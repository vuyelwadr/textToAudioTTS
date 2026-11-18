#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

KYUTAI_DIR="$MAHTI_EXTERNAL_ROOT/delayed-streams-modeling"
if [[ -f "$KYUTAI_DIR/scripts/tts_pytorch.py" ]]; then
	:
elif command -v git >/dev/null 2>&1; then
	ensure_repo "https://github.com/kyutai-labs/delayed-streams-modeling.git" "$KYUTAI_DIR"
elif [[ ! -f "$KYUTAI_DIR/scripts/tts_pytorch.py" ]]; then
	ARCHIVE_DIR="$MAHTI_EXTERNAL_ROOT/delayed-streams-modeling-main"
	rm -rf "$ARCHIVE_DIR" "$KYUTAI_DIR"
	download_tar_gz_to_dir \
		"https://github.com/kyutai-labs/delayed-streams-modeling/archive/refs/heads/main.tar.gz" \
		"$MAHTI_EXTERNAL_ROOT" \
		"$ARCHIVE_DIR/scripts/tts_pytorch.py"
	[[ -d "$ARCHIVE_DIR" ]] || die "Kyutai archive extraction failed: $ARCHIVE_DIR"
	mv "$ARCHIVE_DIR" "$KYUTAI_DIR"
fi

load_pytorch_module 2.9
ensure_python_venv "kyutai-tts"
activate_python_venv "kyutai-tts"

# The xet fast-transfer protocol issues one API call per chunk, which blows
# through HF's 1000-requests/5-minutes quota on a large multi-file snapshot
# download (moshi's TTS repo + the sizeable voices repo). Force plain HTTP
# downloads instead, which use far fewer requests and avoid that quota.
export HF_HUB_DISABLE_XET=1

python -m pip install --no-deps "moshi==0.2.11"
python -m pip install \
	"aiohttp<3.12,>=3.10.5" \
	"bitsandbytes<0.46,>=0.45" \
	"huggingface-hub<0.34,>=0.24" \
	"safetensors<0.6,>=0.4.0" \
	"sentencepiece==0.2" \
	"sounddevice==0.5" \
	"sphn<0.2" \
	soundfile
python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO

model_root = Path(os.environ["MAHTI_MODEL_ROOT"]) / "kyutai"
model_root.mkdir(parents=True, exist_ok=True)

snapshot_download(
	repo_id=DEFAULT_DSM_TTS_REPO,
	local_dir=str(model_root / "model"),
	local_dir_use_symlinks=False,
)
snapshot_download(
	repo_id=DEFAULT_DSM_TTS_VOICE_REPO,
	local_dir=str(model_root / "voices"),
	local_dir_use_symlinks=False,
)
PY

log "Kyutai TTS bootstrap completed"