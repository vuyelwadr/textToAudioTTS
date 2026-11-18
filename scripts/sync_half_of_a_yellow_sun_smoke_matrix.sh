#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

LOCAL_ROOT="${LOCAL_ROOT:-$REPO_ROOT/generated/audiobooks-half_of_a_yellow_sun-smoke-matrix}"
MAHTI_REMOTE="${MAHTI_REMOTE:-/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/qwen/}"
PUHTI_REMOTE="${PUHTI_REMOTE:-/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-smoke-matrix/vibevoice/}"

mkdir -p "$LOCAL_ROOT/qwen" "$LOCAL_ROOT/vibevoice"

rsync -az mahti.csc.fi:"$MAHTI_REMOTE" "$LOCAL_ROOT/qwen/"
rsync -az puhti.csc.fi:"$PUHTI_REMOTE" "$LOCAL_ROOT/vibevoice/"

printf 'Synced smoke matrix into %s\n' "$LOCAL_ROOT"
