#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.7
ensure_python_venv "gpt-sovits"

GPT_DIR="$MAHTI_EXTERNAL_ROOT/GPT-SoVITS"
ensure_repo "https://github.com/RVC-Boss/GPT-SoVITS.git" "$GPT_DIR"
activate_python_venv "gpt-sovits"

GPT_REQUIREMENTS_FILTERED="$TMPDIR/gpt-sovits-requirements.filtered.txt"

python -m pip install cmake ninja
python -m pip install opencc-python-reimplemented
python -m pip install -r "$GPT_DIR/extra-req.txt" --no-deps
grep -vE '^(--no-binary=opencc|opencc)$' "$GPT_DIR/requirements.txt" > "$GPT_REQUIREMENTS_FILTERED"
python -m pip install -r "$GPT_REQUIREMENTS_FILTERED"
python -m pip install faster-whisper

if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "ffmpeg was not found in PATH; GPT-SoVITS preprocessing may fail until ffmpeg is available"
fi

download_zip_to_dir \
    "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip" \
    "$GPT_DIR/GPT_SoVITS" \
    "$GPT_DIR/GPT_SoVITS/pretrained_models/sv"

python - <<'PY'
import os
import pyopenjtalk
import sys

print(sys.prefix)
print(os.path.dirname(pyopenjtalk.__file__))
PY

PY_PREFIX=$(python - <<'PY'
import sys
print(sys.prefix)
PY
)

PYOPENJTALK_PREFIX=$(python - <<'PY'
import os
import pyopenjtalk
print(os.path.dirname(pyopenjtalk.__file__))
PY
)

download_zip_to_dir \
    "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip" \
    "$PY_PREFIX" \
    "$PY_PREFIX/nltk_data"

download_tar_gz_to_dir \
    "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz" \
    "$PYOPENJTALK_PREFIX" \
    "$PYOPENJTALK_PREFIX/open_jtalk_dic_utf_8-1.11"

log "GPT-SoVITS bootstrap completed"