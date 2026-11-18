#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QWEN_TIMING_JSON="generated/audiobooks-half_of_a_yellow_sun-full/qwen3_faster_clone_active_line_audited_mac1080.timing.json"
QWEN_REMOTE_MP4="/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/qwen3_faster_clone_active_line_audited_mac1080.mp4"
QWEN_LOCAL_MP4="generated/audiobooks-half_of_a_yellow_sun-full/qwen3_faster_clone_active_line_audited_mac1080.mp4"
QWEN_CHUNK_DIR="generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/qwen3_faster_clone/chunks"
QWEN_ASR_CACHE="generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/qwen3_faster_clone/chunk_audit/asr_cache"
QWEN_TEMP_GLOB_1="autoscroll_video_*"
QWEN_TEMP_GLOB_2="generated/audiobooks-half_of_a_yellow_sun-full/autoscroll_video_*"
ORCH_LOG="generated/audiobooks-half_of_a_yellow_sun-full/audit_logs/orchestrate_qwen_copy_then_vibe.log"
QWEN_COPY_DONE="generated/audiobooks-half_of_a_yellow_sun-full/qwen_copy_done.marker"
VIBE_REMOTE_CHUNK_DIR="/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/vibevoice_realtime/chunks"
VIBE_LOCAL_CHUNK_DIR="$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/vibevoice_realtime/chunks"

mkdir -p "$(dirname "$ORCH_LOG")"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$ORCH_LOG"
}

available_bytes() {
  df -k . | awk 'NR==2 {print $4 * 1024}'
}

log "waiting for qwen timing json"
while [[ ! -f "$QWEN_TIMING_JSON" ]]; do
  sleep 30
done

log "qwen timing json detected; waiting for active qwen render processes to exit"
while pgrep -af 'make_autoscroll_video.py.*qwen3_faster_clone' >/dev/null 2>&1 || \
      pgrep -af 'ffmpeg.*qwen3_faster_clone_stream_fifo.wav' >/dev/null 2>&1; do
  sleep 10
done

log "pruning qwen-only local render artifacts before attempting copy"
rm -rf "$QWEN_CHUNK_DIR" "$QWEN_ASR_CACHE"
rm -f $QWEN_TEMP_GLOB_1 $QWEN_TEMP_GLOB_2 2>/dev/null || true

REMOTE_SIZE="$(ssh mahti "stat -c %s '$QWEN_REMOTE_MP4' 2>/dev/null || stat -f %z '$QWEN_REMOTE_MP4'")"
FREE_BYTES="$(available_bytes)"
MARGIN_BYTES=$((512 * 1024 * 1024))

log "remote qwen mp4 size=${REMOTE_SIZE} bytes free_local=${FREE_BYTES} bytes"

if (( FREE_BYTES > REMOTE_SIZE + MARGIN_BYTES )); then
  log "enough local space; copying qwen mp4 first"
  rsync -av --append-verify "mahti:$QWEN_REMOTE_MP4" "$QWEN_LOCAL_MP4"
  if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -v error -show_entries format=duration,size -of json "$QWEN_LOCAL_MP4" | tee -a "$ORCH_LOG"
  fi
  touch "$QWEN_COPY_DONE"
  log "qwen copy complete"
else
  log "not enough local space to copy qwen first; skipping copy for now"
fi

if [[ ! -d "$VIBE_LOCAL_CHUNK_DIR" ]]; then
  log "vibevoice local chunk dir missing; restaging from mahti"
  mkdir -p "$(dirname "$VIBE_LOCAL_CHUNK_DIR")"
  rsync -av --delete "mahti:$VIBE_REMOTE_CHUNK_DIR/" "$VIBE_LOCAL_CHUNK_DIR/"
  log "vibevoice chunk restage complete"
fi

log "starting vibevoice full verified render"
exec "$ROOT_DIR/scripts/run_local_audited_video_to_remote.sh" \
  --model vibevoice_realtime \
  --audit-summary "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/vibevoice_realtime/chunk_audit/audit_summary.json" \
  --chunk-text-dir "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/chunks" \
  --chunk-audio-dir "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/vibevoice_realtime/chunks" \
  --remote-audio "/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime.wav" \
  --remote-output "/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime_active_line_audited_mac1080.mp4" \
  --local-fifo "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime_stream_fifo.wav" \
  --stderr-log "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/audit_logs/vibevoice_full_verified_render.log" \
  --timing-json "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime_active_line_audited_mac1080.timing.json" \
  --asr-cache-dir "$ROOT_DIR/generated/audiobooks-half_of_a_yellow_sun-full/half_of_a_yellow_sun/vibevoice_realtime/chunk_audit/asr_cache"
