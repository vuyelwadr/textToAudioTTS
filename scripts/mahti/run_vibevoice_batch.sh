#!/bin/bash
#SBATCH --job-name=vibebatch
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

REPO_ROOT="/scratch/project_2010549/ruwododa/textToAudioTTS-puhti"
cd "$REPO_ROOT"

source venvs/vibevoice/bin/activate

python scripts/mahti/run_vibevoice_batch.py 2>&1