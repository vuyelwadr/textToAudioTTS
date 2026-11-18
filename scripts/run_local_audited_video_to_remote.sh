#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.conda/bin/python"
VIDEO_SCRIPT="$ROOT_DIR/scripts/make_autoscroll_video.py"
FONT_PATH="/System/Library/Fonts/Supplemental/Georgia.ttf"

MODEL=""
CHUNK_TEXT_DIR=""
CHUNK_AUDIO_DIR=""
AUDIT_SUMMARY=""
REMOTE_AUDIO=""
REMOTE_OUTPUT=""
LOCAL_FIFO=""
STDERR_LOG=""
TIMING_JSON=""
ASR_CACHE_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --chunk-text-dir) CHUNK_TEXT_DIR="$2"; shift 2 ;;
    --chunk-audio-dir) CHUNK_AUDIO_DIR="$2"; shift 2 ;;
    --audit-summary) AUDIT_SUMMARY="$2"; shift 2 ;;
    --remote-audio) REMOTE_AUDIO="$2"; shift 2 ;;
    --remote-output) REMOTE_OUTPUT="$2"; shift 2 ;;
    --local-fifo) LOCAL_FIFO="$2"; shift 2 ;;
    --stderr-log) STDERR_LOG="$2"; shift 2 ;;
    --timing-json) TIMING_JSON="$2"; shift 2 ;;
    --asr-cache-dir) ASR_CACHE_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$CHUNK_AUDIO_DIR" || -z "$REMOTE_AUDIO" || -z "$REMOTE_OUTPUT" || -z "$LOCAL_FIFO" || -z "$STDERR_LOG" || -z "$TIMING_JSON" || -z "$ASR_CACHE_DIR" ]]; then
  echo "Missing required arguments" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$VIDEO_SCRIPT" ]]; then
  echo "Missing video script: $VIDEO_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$CHUNK_AUDIO_DIR" ]]; then
  echo "Missing chunk audio dir: $CHUNK_AUDIO_DIR" >&2
  exit 1
fi

if [[ -n "$AUDIT_SUMMARY" ]]; then
  if [[ ! -f "$AUDIT_SUMMARY" ]]; then
    echo "Missing audit summary: $AUDIT_SUMMARY" >&2
    exit 1
  fi
  AUDIT_CHECK="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
payload = json.loads(Path("$AUDIT_SUMMARY").read_text(encoding="utf-8"))
print(payload.get("blocking_issue_count", 0))
print(payload.get("corrected_text_dir", ""))
PY
)"
  BLOCKING_COUNT="$(printf '%s\n' "$AUDIT_CHECK" | sed -n '1p')"
  CORRECTED_TEXT_DIR="$(printf '%s\n' "$AUDIT_CHECK" | sed -n '2p')"
  if [[ "$BLOCKING_COUNT" != "0" ]]; then
    echo "Audit summary reports blocking issues for $MODEL; refusing full render" >&2
    exit 1
  fi
  if [[ -n "$CORRECTED_TEXT_DIR" && -d "$CORRECTED_TEXT_DIR" ]]; then
    CHUNK_TEXT_DIR="$CORRECTED_TEXT_DIR"
  fi
fi

if [[ -z "$CHUNK_TEXT_DIR" || ! -d "$CHUNK_TEXT_DIR" ]]; then
  echo "Missing usable chunk text dir: $CHUNK_TEXT_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$STDERR_LOG")" "$(dirname "$TIMING_JSON")" "$ASR_CACHE_DIR"
rm -f "$LOCAL_FIFO"
mkfifo "$LOCAL_FIFO"

cleanup() {
  if [[ -n "${SSH_CAT_PID:-}" ]]; then
    kill "$SSH_CAT_PID" 2>/dev/null || true
  fi
  rm -f "$LOCAL_FIFO"
}
trap cleanup EXIT

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
START_EPOCH="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"

echo "[$(date -Iseconds)] streaming $MODEL audio from mahti" >> "$STDERR_LOG"
ssh mahti "cat '$REMOTE_AUDIO'" > "$LOCAL_FIFO" &
SSH_CAT_PID=$!

set -o pipefail
echo "[$(date -Iseconds)] rendering $MODEL locally and streaming video to mahti:$REMOTE_OUTPUT" >> "$STDERR_LOG"
"$PYTHON_BIN" "$VIDEO_SCRIPT" \
  --audio "$LOCAL_FIFO" \
  --chunk-text-dir "$CHUNK_TEXT_DIR" \
  --chunk-audio-dir "$CHUNK_AUDIO_DIR" \
  --output - \
  --width 1920 \
  --height 1080 \
  --fps 24 \
  --video-codec libx264 \
  --video-preset veryfast \
  --video-crf 18 \
  --font "$FONT_PATH" \
  --paragraph-size 56 \
  --heading-size 88 \
  --focus-line-ratio 0.62 \
  --show-active-line \
  --active-line-timing asr \
  --active-line-asr-model small.en \
  --active-line-asr-device cpu \
  --active-line-asr-workers auto \
  --active-line-asr-cache-dir "$ASR_CACHE_DIR" \
  2>>"$STDERR_LOG" \
  | ssh mahti "cat > '$REMOTE_OUTPUT'"
set +o pipefail

wait "$SSH_CAT_PID"

FINISHED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
END_EPOCH="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"

AUDIO_DURATION="$("$PYTHON_BIN" - <<PY
from pathlib import Path
from scripts.make_autoscroll_video import probe_audio_duration
print(probe_audio_duration(Path("$REMOTE_AUDIO")))
PY
 2>/dev/null || ssh mahti "ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 '$REMOTE_AUDIO'")"

REMOTE_SIZE="$(ssh mahti "stat -c %s '$REMOTE_OUTPUT' 2>/dev/null || stat -f %z '$REMOTE_OUTPUT'")"

cat > "$TIMING_JSON" <<EOF
{
  "model": "$MODEL",
  "started_at": "$STARTED_AT",
  "finished_at": "$FINISHED_AT",
  "elapsed_seconds": $("$PYTHON_BIN" - <<PY
start = float("$START_EPOCH")
end = float("$END_EPOCH")
print(f"{end - start:.6f}")
PY
),
  "audio_duration_seconds": $AUDIO_DURATION,
  "remote_audio": "$REMOTE_AUDIO",
  "remote_output": "$REMOTE_OUTPUT",
  "remote_size_bytes": $REMOTE_SIZE,
  "chunk_text_dir": "$CHUNK_TEXT_DIR",
  "chunk_audio_dir": "$CHUNK_AUDIO_DIR",
  "audit_summary": "$AUDIT_SUMMARY",
  "asr_cache_dir": "$ASR_CACHE_DIR",
  "font": "$FONT_PATH",
  "width": 1920,
  "height": 1080,
  "fps": 24,
  "video_codec": "libx264",
  "video_preset": "veryfast",
  "video_crf": 18,
  "active_line_timing": "asr"
}
EOF

echo "[$(date -Iseconds)] completed $MODEL remote render ($REMOTE_SIZE bytes)" >> "$STDERR_LOG"
