#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

load_pytorch_module 2.9
ensure_python_venv "openvoice-v2"

OPENVOICE_DIR="$MAHTI_EXTERNAL_ROOT/OpenVoice"
MELO_DIR="$MAHTI_EXTERNAL_ROOT/MeloTTS"
ensure_repo "https://github.com/myshell-ai/OpenVoice.git" "$OPENVOICE_DIR"
ensure_repo "https://github.com/myshell-ai/MeloTTS.git" "$MELO_DIR"
activate_python_venv "openvoice-v2"

python -m pip install \
    "numpy<2" \
    anyascii \
    cached_path \
    cn2an \
    eng_to_ipa \
    faster-whisper \
    fugashi \
    g2p_en \
    huggingface_hub \
    inflect \
    jieba \
    langid \
    librosa \
    loguru \
    mecab-python3 \
    num2words \
    pydub \
    pykakasi \
    pypinyin \
    "gruut[de,es,fr]==2.2.3" \
    jamo \
    soundfile \
    tqdm \
    transformers \
    txtsplit \
    unidic==1.1.0 \
    unidecode \
    unidic_lite \
    wavmark \
    whisper-timestamped
python -m pip install --no-deps -e "$OPENVOICE_DIR"
python -m unidic download
python - <<'PY'
import nltk

for package in ("averaged_perceptron_tagger_eng", "cmudict"):
    nltk.download(package, quiet=True)
PY

download_zip_to_dir \
    "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip" \
    "$OPENVOICE_DIR" \
    "$OPENVOICE_DIR/checkpoints_v2/converter/checkpoint.pth"

log "OpenVoice V2 bootstrap completed"