#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

REMOTE_ROOT="/scratch/project_2010549/ruwododa/textToAudioTTS-mahti/generated/inworld_tts_training"
LOCAL_ROOT="$REPO_ROOT/generated/inworld_tts_training"
LOG_DIR="$LOCAL_ROOT/watch_logs"
mkdir -p "$LOG_DIR"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_FILE"
}

sync_run_artifacts() {
    local run_name="$1"
    local remote_run="$REMOTE_ROOT/$run_name/"
    local local_run="$LOCAL_ROOT/$run_name/"

    mkdir -p "$local_run"
    rsync -az \
        --include '*/' \
        --include 'summary.json' \
        --include 'outputs/***' \
        --exclude '*' \
        "mahti:$remote_run" "$local_run" >>"$LOG_FILE" 2>&1 || true
}

job_terminal_state() {
    local state="$1"
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [[ $# -lt 2 || $(( $# % 2 )) -ne 0 ]]; then
    echo "usage: $0 <job-id> <run-name> [<job-id> <run-name> ...]" >&2
    exit 2
fi

declare -a JOB_IDS=()
declare -a RUN_NAMES=()
declare -a LAST_STATES=()

while [[ $# -gt 0 ]]; do
    job_id="$1"
    run_name="$2"
    JOB_IDS+=("$job_id")
    RUN_NAMES+=("$run_name")
    LAST_STATES+=("UNKNOWN")
    shift 2
done

LOG_FILE="$LOG_DIR/inworld_watch_$(date '+%Y%m%d_%H%M%S').log"
touch "$LOG_FILE"
log "watching jobs: ${JOB_IDS[*]}"

while true; do
    all_terminal=1
    for i in "${!JOB_IDS[@]}"; do
        job_id="${JOB_IDS[$i]}"
        run_name="${RUN_NAMES[$i]}"
        state="$(ssh mahti "sacct -j $job_id --format=State -n | head -n 1 | awk '{print \$1}'" 2>/dev/null || echo UNKNOWN)"
        state="${state%%+*}"
        [[ -n "$state" ]] || state="UNKNOWN"

        if [[ "${LAST_STATES[$i]}" != "$state" ]]; then
            log "job $job_id ($run_name) state: $state"
            LAST_STATES[$i]="$state"
        fi

        sync_run_artifacts "$run_name"

        if ! job_terminal_state "$state"; then
            all_terminal=0
        fi
    done

    if [[ "$all_terminal" -eq 1 ]]; then
        log "all watched jobs reached terminal states"
        break
    fi

    sleep 60
done

for i in "${!JOB_IDS[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    sync_run_artifacts "$run_name"
done

log "final sync complete"
