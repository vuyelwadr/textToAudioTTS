#!/bin/bash
# Lightweight poller that auto-submits the two Prometheus Qwen3 jobs
# once Mahti's pending-job count drops below a safe threshold.
#
# Usage (run on a Mahti login node):
#   nohup bash scripts/mahti/poll_and_submit_prometheus.sh > poll_prometheus.log 2>&1 &
#   disown
#
# To stop the poller later:
#   pkill -f poll_and_submit_prometheus

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Config
PENDING_THRESHOLD=5          # Submit when pending jobs <= this
POLL_INTERVAL_MINUTES=10     # Check every N minutes
JOB_A="scripts/mahti/run_prometheus_atlas_qwen3_optA.sh"
JOB_B="scripts/mahti/run_prometheus_atlas_qwen3_optB.sh"
SAFE_JOB_LIMIT=10            # Hard ceiling; if pending > this we never submit

echo "[poller] Started at $(date -Is)"
echo "[poller] Repo root: $REPO_ROOT"
echo "[poller] Will submit when pending jobs <= $PENDING_THRESHOLD"
echo "[poller] Polling every ${POLL_INTERVAL_MINUTES}m"

while true; do
    # Count pending jobs for current user
    PENDING_COUNT=$(squeue -u "$USER" -t PD -h 2>/dev/null | wc -l | tr -d ' ')
    # Also count running to make sure we aren't flooding
    RUNNING_COUNT=$(squeue -u "$USER" -t R -h 2>/dev/null | wc -l | tr -d ' ')

    echo "[poller] $(date -Is) pending=$PENDING_COUNT running=$RUNNING_COUNT"

    # Check if our target jobs are already queued (by name prefix)
    OPTA_QUEUED=$(squeue -u "$USER" -n prom-qwen-optA -h 2>/dev/null | wc -l | tr -d ' ')
    OPTB_QUEUED=$(squeue -u "$USER" -n prom-qwen-optB -h 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$OPTA_QUEUED" -gt 0 && "$OPTB_QUEUED" -gt 0 ]]; then
        echo "[poller] Both jobs already in queue. Exiting."
        exit 0
    fi

    if [[ "$PENDING_COUNT" -le "$PENDING_THRESHOLD" && "$PENDING_COUNT" -lt "$SAFE_JOB_LIMIT" ]]; then
        echo "[poller] Threshold met. Submitting jobs..."
        cd "$REPO_ROOT" || exit 1

        if [[ "$OPTA_QUEUED" -eq 0 ]]; then
            echo "[poller] Submitting Option A..."
            sbatch "$JOB_A"
        fi

        if [[ "$OPTB_QUEUED" -eq 0 ]]; then
            echo "[poller] Submitting Option B..."
            sbatch "$JOB_B"
        fi

        # Verify they got in
        sleep 2
        NEW_OPTA=$(squeue -u "$USER" -n prom-qwen-optA -h 2>/dev/null | wc -l | tr -d ' ')
        NEW_OPTB=$(squeue -u "$USER" -n prom-qwen-optB -h 2>/dev/null | wc -l | tr -d ' ')

        if [[ "$NEW_OPTA" -gt 0 && "$NEW_OPTB" -gt 0 ]]; then
            echo "[poller] Both jobs successfully queued. Exiting."
            exit 0
        else
            echo "[poller] Submission may have failed (AssocMaxSubmitJobLimit?). Will retry next cycle."
        fi
    else
        echo "[poller] Threshold NOT met. Sleeping ${POLL_INTERVAL_MINUTES}m..."
    fi

    sleep "${POLL_INTERVAL_MINUTES}m"
done
