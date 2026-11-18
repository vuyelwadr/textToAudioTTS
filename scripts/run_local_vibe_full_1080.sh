#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full"
FIFO_PATH="$OUT_DIR/Half_Of_A_Yellow_Sun_vibevoice_realtime.wav"
OUTPUT_PATH="$OUT_DIR/vibevoice_realtime_active_line_local_mac1080.mp4"
LOG_PATH="$OUT_DIR/video_logs/vibevoice_realtime_local_render.stderr.log"
TIME_PATH="$OUT_DIR/vibevoice_realtime_active_line_local_stream.timing.txt"
TMP_PATH="$OUT_DIR/.tmp_vibe_render"
TEXT_DIR="$OUT_DIR/half_of_a_yellow_sun/chunks"
DURATION_MANIFEST="$OUT_DIR/vibevoice_realtime_chunk_durations.json"
FONT_PATH="/System/Library/Fonts/Supplemental/Georgia.ttf"
REMOTE_WAV="/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime.wav"
CHUNK_AUDIO_DIR="$OUT_DIR/half_of_a_yellow_sun/vibevoice_realtime/chunks"
ASR_CACHE_DIR="$OUT_DIR/vibevoice_realtime_active_line_asr_cache"
CORRECTED_TEXT_DIR="$OUT_DIR/half_of_a_yellow_sun/vibevoice_realtime/chunk_audit/corrected_chunk_text"

if [[ -d "$CORRECTED_TEXT_DIR" ]]; then
  TEXT_DIR="$CORRECTED_TEXT_DIR"
fi

mkdir -p "$OUT_DIR/video_logs" "$TMP_PATH"
rm -f "$FIFO_PATH" "$OUTPUT_PATH" "$LOG_PATH" "$TIME_PATH"
mkfifo "$FIFO_PATH"

if [[ ! -d "$CHUNK_AUDIO_DIR" ]]; then
  echo "Missing local chunk audio dir for ASR-synced active-line render: $CHUNK_AUDIO_DIR" >&2
  echo "Refusing to render with chunk-proportional timing because it produces visibly wrong sync." >&2
  exit 1
fi

AUDIT_SUMMARY="$OUT_DIR/half_of_a_yellow_sun/vibevoice_realtime/chunk_audit/audit_summary.json"
if [[ -f "$AUDIT_SUMMARY" ]]; then
  if "$ROOT_DIR/.conda/bin/python" - <<PY
import json
from pathlib import Path
summary = json.loads(Path("$AUDIT_SUMMARY").read_text(encoding="utf-8"))
raise SystemExit(1 if summary.get("blocking_issue_count", 0) else 0)
PY
  then
    :
  else
    echo "Chunk audit reported blocking issues for vibevoice_realtime; refusing to render full video." >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${SSH_CAT_PID:-}" ]]; then
    kill "$SSH_CAT_PID" 2>/dev/null || true
  fi
  rm -f "$FIFO_PATH"
}
trap cleanup EXIT

echo "[$(date -Iseconds)] streaming audio for vibevoice_realtime from mahti into local fifo" >> "$LOG_PATH"
ssh mahti cat "$REMOTE_WAV" > "$FIFO_PATH" &
SSH_CAT_PID=$!

export TMPDIR="$TMP_PATH"
echo "[$(date -Iseconds)] rendering vibevoice_realtime locally to $OUTPUT_PATH" >> "$LOG_PATH"
/usr/bin/time -p -o "$TIME_PATH" \
  "$ROOT_DIR/.conda/bin/python" "$ROOT_DIR/scripts/make_autoscroll_video.py" \
  --chunk-text-dir "$TEXT_DIR" \
  --chunk-audio-dir "$CHUNK_AUDIO_DIR" \
  --audio "$FIFO_PATH" \
  --output "$OUTPUT_PATH" \
  --show-active-line \
  --active-line-timing asr \
  --active-line-asr-model small.en \
  --active-line-asr-device cpu \
  --active-line-asr-cache-dir "$ASR_CACHE_DIR" \
  --font "$FONT_PATH" \
  --width 1920 \
  --height 1080 \
  --fps 24 \
  --paragraph-size 56 \
  --heading-size 88 \
  --focus-line-ratio 0.62 \
  --video-codec libx264 \
  --video-preset veryfast \
  --video-crf 18 \
  2>> "$LOG_PATH"

wait "$SSH_CAT_PID"
echo "[$(date -Iseconds)] finished vibevoice_realtime local render" >> "$LOG_PATH"
