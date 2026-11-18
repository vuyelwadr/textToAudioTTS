#!/usr/bin/env bash
# Bootstrap script for Higgs Audio v3 TTS (bosonai/higgs-audio-v3-tts-4b).
#
# Unlike v2, the v3 weights repo ships NO modeling code: it is config +
# safetensors + tokenizer only (architecture "HiggsMultimodalQwen3ForConditionalGeneration",
# model_type "higgs_multimodal_qwen3"). That class does not exist in
# transformers as of 5.13.1 (only `higgs_audio_v2` is upstreamed there), and
# there is no trust_remote_code fallback either since the HF repo has no .py
# file to load. Boson's own model card / AGENTS.md only document two serving
# stacks: SGLang-Omni (Docker container) or vLLM-Omni.
#
# vLLM-Omni is the lighter of the two for a one-off test clip: it's a plain
# `pip install`, needs no Docker/root, and ships an *offline* batch-inference
# example (no HTTP server) -- see run_higgs_audio_v3.py. SGLang-Omni is
# server-only in its documented usage and would be overkill here.
#
# Voice cloning additionally needs a *codec* checkpoint to encode the
# reference clip. The official `bosonai/higgs-audio-v2-tokenizer` repo's
# model.safetensors is mislabeled (it's actually a copy of the 3B talker LM,
# not the codec -- confirmed by reading vllm_omni's higgs_audio_v2_tokenizer.py
# source), so vLLM-Omni instead pulls the working codec weights from a
# third-party repackage: k2-fsa/OmniVoice (audio_tokenizer/ subdir, ~800MB).
# That code resolves repos via the *standard* huggingface_hub cache layout
# (models--org--repo/snapshots/<rev>/...), not an arbitrary local_dir, so we
# pre-warm both repos into the standard HF cache here instead of using this
# repo's usual python_hf_download-into-local_dir helper.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"

_bootstrap() {
	load_pytorch_module 2.9
	ensure_python_venv "higgs-audio-v3"
	activate_python_venv "higgs-audio-v3"

	# vLLM core is not declared as a formal vllm-omni dependency (vllm-omni
	# registers itself as a "vllm.general_plugins" entry point and expects
	# vllm to already be present), so install both explicitly. No tight pin:
	# Boson's recipe tracks a moving vllm-omni feature branch/commit, not a
	# released vllm version, so we take current PyPI and let the resolver
	# reuse the module-provided torch if the pins allow it.
	python -m pip install \
		vllm \
		vllm-omni \
		soundfile \
		librosa \
		jinja2

	log "Pre-warming standard HF cache for higgs-audio-v3-tts-4b and the k2-fsa/OmniVoice codec"
	python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(repo_id="bosonai/higgs-audio-v3-tts-4b")
snapshot_download(repo_id="k2-fsa/OmniVoice", allow_patterns=["audio_tokenizer/*"])
PY

	log "Higgs Audio v3 bootstrap completed"
}

run_bootstrap_once "higgs-audio-v3" _bootstrap
activate_python_venv "higgs-audio-v3"
log "Higgs Audio v3 environment ready"
