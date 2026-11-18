#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.conda/bin/python"
VIDEO_SCRIPT="$ROOT_DIR/scripts/make_autoscroll_video.py"
BOOK_DIR="$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full"
CHUNK_TEXT_DIR="$BOOK_DIR/half_of_a_yellow_sun/chunks"
LOG_DIR="$BOOK_DIR/video_logs"
mkdir -p "$LOG_DIR"

REMOTE_ROOT="/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full"
FONT_PATH="/System/Library/Fonts/Supplemental/Georgia.ttf"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi

run_render() {
  local model="$1"
  local local_audio="$2"
  local remote_audio="$3"
  local duration_manifest="$4"
  local remote_output="$5"
  local chunk_audio_dir="$6"
  local stderr_log="$LOG_DIR/${model}_local_render.stderr.log"
  local timing_json="$BOOK_DIR/${model}_active_line_local_stream.timing.json"
  local asr_cache_dir="$BOOK_DIR/${model}_active_line_asr_cache"
  local corrected_text_dir="$BOOK_DIR/half_of_a_yellow_sun/${model}/chunk_audit/corrected_chunk_text"
  local chunk_text_dir="$CHUNK_TEXT_DIR"
  local audit_summary="$BOOK_DIR/half_of_a_yellow_sun/${model}/chunk_audit/audit_summary.json"

  if [[ -d "$corrected_text_dir" ]]; then
    chunk_text_dir="$corrected_text_dir"
  fi

  if [[ ! -d "$chunk_audio_dir" ]]; then
    echo "Missing local chunk audio dir for ${model}: ${chunk_audio_dir}" >&2
    echo "Refusing to render ${model} with chunk-proportional timing because it produces visibly wrong sync." >&2
    exit 1
  fi

  if [[ -f "$audit_summary" ]]; then
    if "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
summary = json.loads(Path("$audit_summary").read_text(encoding="utf-8"))
raise SystemExit(1 if summary.get("blocking_issue_count", 0) else 0)
PY
    then
      :
    else
      echo "Chunk audit reported blocking issues for ${model}; refusing to render full video." >&2
      exit 1
    fi
  fi

  local audio_duration
  audio_duration="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
entries = json.loads(Path("${duration_manifest}").read_text(encoding="utf-8"))
print(sum(float(item["duration_seconds"]) for item in entries))
PY
)"

  rm -f "$local_audio"
  mkfifo "$local_audio"

  echo "[$(date -Iseconds)] streaming audio for ${model} from mahti into local fifo" | tee -a "$stderr_log"
  ssh mahti "cat '$remote_audio'" > "$local_audio" &
  local audio_stream_pid=$!

  echo "[$(date -Iseconds)] rendering ${model} locally and streaming video to mahti:${remote_output}" | tee -a "$stderr_log"
  local started_at finished_at elapsed_seconds remote_size
  started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  local start_epoch
  start_epoch="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"

  set -o pipefail
  /usr/bin/time -p "$PYTHON_BIN" "$VIDEO_SCRIPT" \
      --audio "$local_audio" \
      --chunk-text-dir "$chunk_text_dir" \
      --chunk-audio-dir "$chunk_audio_dir" \
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
      --active-line-asr-cache-dir "$asr_cache_dir" \
      2>>"$stderr_log" \
      | ssh mahti "cat > '$remote_output'"

  local end_epoch
  end_epoch="$("$PYTHON_BIN" - <<'PY'
import time
print(time.time())
PY
)"
  set +o pipefail
  wait "$audio_stream_pid"

  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  elapsed_seconds="$("$PYTHON_BIN" - <<PY
start = float("${start_epoch}")
end = float("${end_epoch}")
print(f"{end - start:.6f}")
PY
)"
  remote_size="$(ssh mahti "stat -c %s '$remote_output'")"

  cat > "$timing_json" <<EOF
{
  "started_at": "${started_at}",
  "finished_at": "${finished_at}",
  "elapsed_seconds": ${elapsed_seconds},
  "audio_duration_seconds": ${audio_duration},
  "speed_x_realtime": $("$PYTHON_BIN" - <<PY
audio = float("${audio_duration}")
elapsed = float("${elapsed_seconds}")
print(audio / elapsed if elapsed else 0.0)
PY
),
  "audio": "${local_audio}",
  "remote_audio": "${remote_audio}",
  "remote_output": "${remote_output}",
  "remote_size_bytes": ${remote_size},
  "chunk_text_dir": "${CHUNK_TEXT_DIR}",
  "chunk_duration_manifest": "${duration_manifest}",
  "chunk_audio_dir": "${chunk_audio_dir}",
  "font": "${FONT_PATH}",
  "video_codec": "libx264",
  "video_preset": "veryfast",
  "video_crf": 18,
  "width": 1920,
  "height": 1080,
  "fps": 24,
  "active_line_timing": "asr"
}
EOF

  echo "[$(date -Iseconds)] completed ${model}, remote size ${remote_size} bytes" | tee -a "$stderr_log"
  rm -f "$local_audio"
}

run_render \
  "vibevoice_realtime" \
  "$BOOK_DIR/Half_Of_A_Yellow_Sun_vibevoice_realtime.wav" \
  "$REMOTE_ROOT/vibevoice_realtime.wav" \
  "$BOOK_DIR/vibevoice_realtime_chunk_durations.json" \
  "$REMOTE_ROOT/vibevoice_realtime_active_line_local_mac1080.mp4" \
  "$BOOK_DIR/half_of_a_yellow_sun/vibevoice_realtime/chunks"

run_render \
  "qwen3_faster_clone" \
  "$BOOK_DIR/Half_Of_A_Yellow_Sun_qwen3_faster_clone.wav" \
  "$REMOTE_ROOT/qwen3_faster_clone.wav" \
  "$BOOK_DIR/qwen3_faster_clone_chunk_durations.json" \
  "$REMOTE_ROOT/qwen3_faster_clone_active_line_local_mac1080.mp4" \
  "$BOOK_DIR/half_of_a_yellow_sun/qwen3_faster_clone/chunks"
