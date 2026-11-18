#!/bin/bash
#SBATCH --job-name=pymander-qwen
#SBATCH --account=project_2010549
#SBATCH --partition=gpumedium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:gh200:2
#SBATCH --mem=0
#SBATCH --output=/scratch/project_2010549/ruwododa/textToAudioTTS-roihu/tasks/hermetica_20261119/slurm-pymander-%j.out
set -euo pipefail
export CSC_CLUSTER_NAME=roihu
REPO=/scratch/project_2010549/ruwododa/textToAudioTTS-roihu
cd "$REPO"
source scripts/mahti/common.sh
ensure_python310_plus
python -c "import torch; v=torch.__version__; print(v); raise SystemExit(0 if v.startswith('2.9.1') else 1)" \
  || python -m pip install --break-system-packages --upgrade "torch==2.9.1" "torchaudio==2.9.1" --index-url https://download.pytorch.org/whl/cu128

REF_DIR=$REPO/outputs/video_downloads/LhKBK6s_lMw
INPUT_FILE=$REPO/library/divinePymander/divinePymander.txt \
BOOK_STEM=divinePymander \
BOOK_SLUG=divine_pymander \
TARGET_AUDIO_SECONDS=0 \
GENERATE_VIDEO=1 \
VIDEO_RENDER_MODE=submit_cpu \
VIDEO_SUBMIT_CLUSTER=roihu \
VIDEO_SUBMIT_RESOURCE_MODE=cpu \
RUN_LABEL=full_20261119_v1 \
CHUNK_SIZE=850 \
MIN_CHUNK_SIZE=200 \
QWEN_MAX_NEW_TOKENS=2048 \
QWEN_MAX_SEQ_LEN=8192 \
QWEN_BATCH_WORKERS=2 \
AUDIT_CHUNKS=1 \
FORCE=1 \
REFERENCE_AUDIO=$REF_DIR/LhKBK6s_lMw_reference_10s.wav \
REFERENCE_TEXT_FILE=$REF_DIR/LhKBK6s_lMw_reference_10s.txt \
bash scripts/mahti/run_qwen_audiobook_video.sh
