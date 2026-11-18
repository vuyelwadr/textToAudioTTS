#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

ensure_module_command || true
module purge >/dev/null 2>&1 || true
module load python-data/3.10-24.04

INWORLD_REPO_URL="https://github.com/inworld-ai/tts.git"
INWORLD_REPO_DIR="$MAHTI_EXTERNAL_ROOT/inworld-tts"
XCODEC_ROOT="$MAHTI_MODEL_ROOT/inworld-tts/xcodec2"
XCODEC_CKPT_DIR="$XCODEC_ROOT/ckpt"
XCODEC_CKPT_FILE="$XCODEC_CKPT_DIR/epoch=4-step=1400000.ckpt"

mkdir -p "$MAHTI_ENV_ROOT/inworld-tts-poc"

sync_inworld_repo() {
    local lock_fd=""
    local status=0

    acquire_lock lock_fd "repo-inworld-tts"
    ensure_repo "$INWORLD_REPO_URL" "$INWORLD_REPO_DIR" "v0.5.0" || status=$?
    release_lock "$lock_fd"
    return "$status"
}

sync_inworld_repo

patch_torchvision_fake_nms_registration() {
    local meta_file="$INWORLD_REPO_DIR/.venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py"

    [[ -f "$meta_file" ]] || return 0

    python - "$meta_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = '''@torch.library.register_fake("torchvision::nms")
def meta_nms(dets, scores, iou_threshold):
    torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
    torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
    torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
    torch._check(
        dets.size(0) == scores.size(0),
        lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
    )
    ctx = torch._custom_ops.get_ctx()
    num_to_keep = ctx.create_unbacked_symint()
    return dets.new_empty(num_to_keep, dtype=torch.long)
'''
patched = '''if torchvision.extension._has_ops():
    @torch.library.register_fake("torchvision::nms")
    def meta_nms(dets, scores, iou_threshold):
        torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
        torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
        torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
        torch._check(
            dets.size(0) == scores.size(0),
            lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
        )
        ctx = torch._custom_ops.get_ctx()
        num_to_keep = ctx.create_unbacked_symint()
        return dets.new_empty(num_to_keep, dtype=torch.long)
'''
if "if torchvision.extension._has_ops():\n    @torch.library.register_fake(\"torchvision::nms\")" in text:
    raise SystemExit(0)
if original not in text:
    raise SystemExit(1)
path.write_text(text.replace(original, patched, 1), encoding="utf-8")
PY
}

patch_inworld_audio_loader_fallback() {
    local data_utils_file="$INWORLD_REPO_DIR/tts/data/data_utils.py"

    [[ -f "$data_utils_file" ]] || return 0

    python - "$data_utils_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

import_block = "import numpy as np\nimport torch\nimport torchaudio\n"
patched_import_block = "import numpy as np\nimport soundfile as sf\nimport torch\nimport torchaudio\n"

original = '''def load_wav(
    wav_path: str, target_sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Loads and preprocesses a WAV file.

    Args:
        wav_path: Path to the WAV file to load
        target_sample_rate: Optional target sample rate to resample to. If None,
            keeps original.

    Returns:
        Tuple of (preprocessed audio tensor with shape [1, num_samples], sample rate)
    """
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sample_rate and sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        sr = target_sample_rate
    return wav, sr
'''

previous_patched = '''def load_wav(
    wav_path: str, target_sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Loads and preprocesses a WAV file.

    Args:
        wav_path: Path to the WAV file to load
        target_sample_rate: Optional target sample rate to resample to. If None,
            keeps original.

    Returns:
        Tuple of (preprocessed audio tensor with shape [1, num_samples], sample rate)
    """
    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as exc:
        logging.warning(
            "torchaudio.load failed for [%s], falling back to soundfile: %s",
            wav_path,
            exc,
        )
        wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sample_rate and sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        sr = target_sample_rate
    return wav, sr
'''

patched = '''def load_wav(
    wav_path: str, target_sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    """Loads and preprocesses a WAV file.

    Args:
        wav_path: Path to the WAV file to load
        target_sample_rate: Optional target sample rate to resample to. If None,
            keeps original.

    Returns:
        Tuple of (preprocessed audio tensor with shape [1, num_samples], sample rate)
    """
    try:
        wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)
    except Exception as exc:
        logging.warning(
            "soundfile read failed for [%s], falling back to torchaudio: %s",
            wav_path,
            exc,
        )
        wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sample_rate and sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
        sr = target_sample_rate
    return wav, sr
'''

if "import soundfile as sf\n" not in text:
    if import_block not in text:
        raise SystemExit(1)
    text = text.replace(import_block, patched_import_block, 1)

if 'logging.warning(\n            "soundfile read failed for [%s], falling back to torchaudio: %s",' in text:
    path.write_text(text, encoding="utf-8")
    raise SystemExit(0)

if previous_patched in text:
    path.write_text(text.replace(previous_patched, patched, 1), encoding="utf-8")
    raise SystemExit(0)

if original not in text:
    raise SystemExit(1)

path.write_text(text.replace(original, patched, 1), encoding="utf-8")
PY
}

patch_inworld_empty_batch_guard() {
    local vectorizer_file="$INWORLD_REPO_DIR/tools/data/data_vectorizer.py"

    [[ -f "$vectorizer_file" ]] || return 0

    python - "$vectorizer_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = '''        if batch is None:
            logging.warning("Skipping batch [%d] because it is empty.", batch_idx)
            continue

        wavs, feats, codes_lengths, batch_samples = batch
'''
patched = '''        if batch is None or batch[0] is None:
            logging.warning("Skipping batch [%d] because it is empty.", batch_idx)
            continue

        wavs, feats, codes_lengths, batch_samples = batch
'''

if "if batch is None or batch[0] is None:" in text:
    raise SystemExit(0)
if original not in text:
    raise SystemExit(1)
path.write_text(text.replace(original, patched, 1), encoding="utf-8")
PY
}

patch_inworld_attention_backend() {
    local modeling_file="$INWORLD_REPO_DIR/tts/core/modeling.py"

    [[ -f "$modeling_file" ]] || return 0

    python - "$modeling_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
original = '_ATTN_IMPLEMENTATION = "flash_attention_2"\n'
patched = '_ATTN_IMPLEMENTATION = "sdpa"\n'

if patched in text:
    raise SystemExit(0)
if original not in text:
    raise SystemExit(1)
path.write_text(text.replace(original, patched, 1), encoding="utf-8")
PY
}

patch_inworld_convert_checkpoint_unwrap() {
    local convert_checkpoint_file="$INWORLD_REPO_DIR/tools/serving/convert_checkpoint.py"

    [[ -f "$convert_checkpoint_file" ]] || return 0

    python - "$convert_checkpoint_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = '    start_time = time.time()\n'
patched = '''    # The model loaded here is already a plain HF module. Forcing a no-op
    # unwrap avoids transformers importing deepspeed during save_pretrained(),
    # which fails in the CSC container wrapper because CUDA_HOME is unset.
    transformers.modeling_utils.unwrap_model = lambda model, *args, **kwargs: model
    start_time = time.time()
'''

if 'transformers.modeling_utils.unwrap_model = lambda model, *args, **kwargs: model' in text:
    raise SystemExit(0)
if needle not in text:
    raise SystemExit(1)
path.write_text(text.replace(needle, patched, 1), encoding="utf-8")
PY
}

ensure_inworld_runtime_python_deps() {
    python - <<'PY'
import importlib.util
import subprocess
import sys

missing = []
for package_name, module_name in (("soundfile", "soundfile"),):
    if importlib.util.find_spec(module_name) is None:
        missing.append(package_name)

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY
}

normalize_inworld_xcodec_config() {
    local config_file="$XCODEC_CKPT_DIR/model_config.json"
    local template_file="$INWORLD_REPO_DIR/example/codec/model_config.json"

    [[ -f "$config_file" ]] || return 0

    python - "$config_file" "$template_file" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
template_path = Path(sys.argv[2])
raw = path.read_text(encoding="utf-8")

if not raw.strip():
    raw = template_path.read_text(encoding="utf-8")

try:
    config = json.loads(raw)
except json.JSONDecodeError:
    config = json.loads(template_path.read_text(encoding="utf-8"))

config.setdefault("model_type", "xcodec2")
config.setdefault("upsample_factors", None)
config.setdefault("kernel_sizes", None)
path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
PY
}

bootstrap_inworld_tts() {
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        python3 -m pip install --user uv
    fi

    if [[ ! -x "$INWORLD_REPO_DIR/.venv/bin/python" ]]; then
        log "Creating Inworld proof-of-concept environment..."
        (
            cd "$INWORLD_REPO_DIR"
            CUDA_VERSION=12.8 ./setup/setup_python.sh
        )
    fi

    # shellcheck disable=SC1091
    source "$INWORLD_REPO_DIR/.venv/bin/activate"

    python_hf_download \
        "HKUSTAudio/xcodec2" \
        "$XCODEC_ROOT" \
        "ckpt/epoch=4-step=1400000.ckpt"

    mkdir -p "$XCODEC_CKPT_DIR"
    cp "$INWORLD_REPO_DIR/example/codec/model_config.json" "$XCODEC_CKPT_DIR/model_config.json"
    normalize_inworld_xcodec_config
}

run_bootstrap_once "inworld-tts-poc" bootstrap_inworld_tts

# shellcheck disable=SC1091
source "$INWORLD_REPO_DIR/.venv/bin/activate"
ensure_inworld_runtime_python_deps
normalize_inworld_xcodec_config
patch_torchvision_fake_nms_registration || warn "Failed to patch torchvision fake NMS registration"
patch_inworld_audio_loader_fallback || warn "Failed to patch Inworld audio loader fallback"
patch_inworld_empty_batch_guard || warn "Failed to patch Inworld empty-batch guard"
patch_inworld_attention_backend || warn "Failed to patch Inworld attention backend"
patch_inworld_convert_checkpoint_unwrap || warn "Failed to patch Inworld convert_checkpoint unwrap handling"

log "Inworld TTS proof-of-concept bootstrap completed"
