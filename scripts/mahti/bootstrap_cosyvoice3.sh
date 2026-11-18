#!/usr/bin/env bash
# Bootstrap script for CosyVoice 3 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512).
#
# CosyVoice has no setup.py/pyproject.toml, so it is not pip-installable;
# run_cosyvoice3.py adds the checked-out repo (plus its vendored
# third_party/Matcha-TTS submodule) to sys.path instead, the same pattern
# already used for Spark-TTS in this repo (see run_spark_tts.py).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# project_2010549's /projappl allocation is already exhausted (see
# bootstrap_higgs_audio.sh for the same workaround), so keep this venv and
# the CosyVoice source checkout on /scratch instead of the default
# /projappl-based project root.
export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
# common.sh itself resolves MAHTI_LOCK_ROOT under the (quota-exhausted)
# /projappl-based project root before this script runs, so even acquiring
# the bootstrap lock file fails unless this is redirected too.
export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
mkdir -p "$MAHTI_ENV_ROOT" "$MAHTI_EXTERNAL_ROOT" "$MAHTI_LOCK_ROOT"

load_pytorch_module 2.9

bootstrap_cosyvoice3() {
    local repo_dir="$MAHTI_EXTERNAL_ROOT/CosyVoice"

    ensure_repo "https://github.com/FunAudioLLM/CosyVoice.git" "$repo_dir"

    # `find ... -print -quit` exits 0 whether or not it printed a match (it
    # only signals traversal *errors*, e.g. a missing path) -- `git submodule`
    # always pre-creates the target directory as an empty placeholder before
    # checkout, so the directory exists but is empty until initialized. The
    # previous `if ! find ... ; then` check on find's exit status was always
    # false in that case, so the submodule fetch below never ran. Check
    # find's *output* instead so an empty (or missing) directory is detected.
    if [[ -z "$(find "$repo_dir/third_party/Matcha-TTS" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        log "fetching Matcha-TTS submodule (third_party/Matcha-TTS)"
        git -C "$repo_dir" submodule update --init --depth 1 third_party/Matcha-TTS
    fi

    ensure_python_venv "cosyvoice3"
    activate_python_venv "cosyvoice3"

    # CosyVoice's own requirements.txt drags in a heavy training/serving stack
    # (deepspeed, tensorrt, vllm, grpc/gradio web-demo deps) plus a hard pin on
    # torch==2.3.1 / torchaudio==2.3.1 that would fight the module-provided
    # PyTorch 2.9 build. None of that is needed for offline zero-shot cloning
    # inference (cosyvoice/cli/*), so install a curated subset instead.
    # Historically CosyVoice's text-normalization path pulled in
    # pynini/WeTextProcessing, which is finicky to build via pip. The current
    # main branch instead falls back to the pure-Python `wetext` package
    # whenever the (proprietary, optional) `ttsfrd` package isn't installed --
    # see cosyvoice/cli/frontend.py -- so no native/onnxruntime build issues
    # are expected here.
    # onnxruntime-gpu's current PyPI wheel links against CUDA 13
    # (libcudart.so.13), which the module-provided PyTorch 2.9 build (CUDA
    # 12.9) doesn't ship, so import fails at process start. CosyVoiceFrontEnd
    # always forces CPUExecutionProvider for the campplus (speaker embedding)
    # session regardless of GPU availability, and onnxruntime falls back to
    # CPU with a warning (not an error) when CUDAExecutionProvider is
    # unavailable for the speech-tokenizer session -- so plain CPU
    # onnxruntime avoids the CUDA-version mismatch entirely. The heavy
    # compute (LLM + flow-matching decoder + HiFT vocoder) all runs through
    # torch directly on GPU either way; only the two small auxiliary ONNX
    # models (speaker xvector, speech tokenizer) are affected.
    # `conformer` (pinned to 0.3.2 in upstream CosyVoice's requirements.txt)
    # backs the vendored Matcha-TTS decoder's `from conformer import
    # ConformerBlock` (third_party/Matcha-TTS/matcha/models/components/decoder.py),
    # which cosyvoice/flow/flow_matching.py imports transitively -- without
    # it, loading the flow-matching decoder config via hyperpyyaml fails with
    # `ModuleNotFoundError: No module named 'conformer'`.
    #
    # `gdown` and `wget` back the same transitive import chain:
    # matcha/utils/utils.py unconditionally does `import gdown` and `import
    # wget` at module level, even though neither is actually invoked on the
    # zero-shot inference path -- without them hyperpyyaml's config load
    # fails with ModuleNotFoundError the same way the missing
    # `matcha`/`conformer` modules did. diffusers/librosa/matplotlib/rich/
    # lightning/hydra-core/pyarrow (matcha's other requirements.txt-listed
    # deps) were all already satisfied transitively in this venv when this
    # was diagnosed, so they are intentionally not repeated here.
    #
    # `pyworld` (pinned to 0.3.4, matching upstream CosyVoice's
    # requirements.txt) backs cosyvoice/dataset/processor.py's module-level
    # `import pyworld as pw` -- hyperpyyaml's config document references a
    # dataset-pipeline object defined there even though the pipeline itself
    # is training-only, so it still gets constructed (and thus imported)
    # when loading configs for zero-shot inference.
    #
    # A full static import scan of cosyvoice/ + third_party/Matcha-TTS/matcha
    # (excluding webui/train/grpc/onnx-export/trt-export files, which the
    # zero-shot CLI inference path never touches) confirmed the *only*
    # remaining unresolved imports beyond this curated list are: `gradio`
    # (matcha/app.py -- a standalone demo script, never imported by anything
    # else), `phonemizer`/`piper_phonemize`/`unidecode` (matcha/text/cleaners.py
    # -- matcha's own text-cleaning pipeline, unused since CosyVoice reuses
    # only matcha's flow-decoder components), `tensorrt`
    # (cosyvoice/utils/file_utils.py -- a function-local import inside
    # convert_onnx_to_trt(), only triggered by explicit TensorRT conversion),
    # and `ttsfrd` (cosyvoice/cli/frontend.py -- wrapped in try/except with
    # an automatic fallback to `wetext`, see the wetext note above). None of
    # those need installing for offline zero-shot cloning inference.
    python -m pip install \
        einops \
        huggingface_hub \
        HyperPyYAML \
        inflect \
        modelscope \
        omegaconf \
        onnxruntime \
        openai-whisper \
        regex \
        scipy \
        soundfile \
        tiktoken \
        transformers \
        wetext \
        x_transformers \
        conformer==0.3.2 \
        gdown \
        wget \
        pyworld==0.3.4

    python_hf_download \
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" \
        "$MAHTI_MODEL_ROOT/cosyvoice3/FunAudioLLM__Fun-CosyVoice3-0.5B-2512"
}

run_bootstrap_once "cosyvoice3" bootstrap_cosyvoice3
activate_python_venv "cosyvoice3"

log "CosyVoice3 bootstrap completed"
