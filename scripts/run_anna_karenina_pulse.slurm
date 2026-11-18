#!/bin/bash
#SBATCH --job-name=anna-karenina-pulse
#SBATCH --account=project_2010549
#SBATCH --partition=gpusmall
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1,nvme:900
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /scratch/project_2010549/ruwododa/textToAudioTTS-mahti

export REPO_ROOT=/scratch/project_2010549/ruwododa/textToAudioTTS-mahti
export REFERENCE_AUDIO="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.wav"
export REFERENCE_TEXT_FILE="$REPO_ROOT/generated/clone_references_2026-04-25/brothersKarmatzov.clone_ref_60s.transcript.txt"

bash scripts/run_anna_karenina_pulse.sh
