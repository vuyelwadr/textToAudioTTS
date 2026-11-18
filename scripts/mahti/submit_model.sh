#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/mahti/submit_model.sh <model-id> [--cluster auto|mahti|puhti|roihu] [--mode smoke|full] [submit options...] [-- runner args...]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL_ID="$1"
shift

detect_cluster_name() {
    local raw_name

    raw_name="${CSC_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-$(hostname 2>/dev/null || echo local)}}"
    raw_name="${raw_name,,}"

    if [[ "$raw_name" == *puhti* ]]; then
        printf '%s\n' "puhti"
        return
    fi
    if [[ "$raw_name" == *mahti* ]]; then
        printf '%s\n' "mahti"
        return
    fi
    if [[ "$raw_name" == *roihu* ]]; then
        printf '%s\n' "roihu"
        return
    fi

    printf '%s\n' "local"
}

resolve_runtime() {
    case "$1" in
        orpheus) printf '%s\n' "01:00:00" ;;
        fish-speech) printf '%s\n' "02:00:00" ;;
        fish-s2pro) printf '%s\n' "03:00:00" ;;
        f5-tts) printf '%s\n' "01:00:00" ;;
        xtts) printf '%s\n' "02:00:00" ;;
        gpt-sovits) printf '%s\n' "02:30:00" ;;
        openvoice-v2) printf '%s\n' "01:00:00" ;;
        omnivoice) printf '%s\n' "02:00:00" ;;
        voxcpm2) printf '%s\n' "02:00:00" ;;
        dia) printf '%s\n' "02:00:00" ;;
        parler-tts-large) printf '%s\n' "02:30:00" ;;
        spark-tts) printf '%s\n' "02:00:00" ;;
        csm-1b) printf '%s\n' "02:00:00" ;;
        qwen3-tts) printf '%s\n' "03:00:00" ;;
        faster-qwen3-tts) printf '%s\n' "02:00:00" ;;
        breeze-tts2) printf '%s\n' "02:00:00" ;;
        cosyvoice3) printf '%s\n' "02:00:00" ;;
        higgs-audio-v3) printf '%s\n' "03:00:00" ;;
        kyutai-tts) printf '%s\n' "01:30:00" ;;
        moss-tts) printf '%s\n' "02:00:00" ;;
        kugelaudio-open) printf '%s\n' "03:00:00" ;;
        echo-tts) printf '%s\n' "03:00:00" ;;
        higgs-audio) printf '%s\n' "03:00:00" ;;
        chatterbox-turbo) printf '%s\n' "01:00:00" ;;
        voxtral) printf '%s\n' "03:00:00" ;;
        vibevoice) printf '%s\n' "02:00:00" ;;
        step-audio-editx) printf '%s\n' "02:00:00" ;;
        inworld-tts-poc) printf '%s\n' "04:00:00" ;;
        *)
            echo "unknown model id: $1" >&2
            exit 2
            ;;
    esac
}

resolve_mem() {
    # Return explicit --mem value for models that need more than the default
    # (Puhti default is 1 GB/CPU; at 10 CPUs that is 10 GB).
    # Return empty string to use the cluster default.
    case "$1" in
        moss-tts) printf '%s\n' "24G" ;;
        kugelaudio-open) printf '%s\n' "48G" ;;
        echo-tts) printf '%s\n' "32G" ;;
        higgs-audio) printf '%s\n' "32G" ;;
        chatterbox-turbo) printf '%s\n' "16G" ;;
        voxtral) printf '%s\n' "48G" ;;
        vibevoice) printf '%s\n' "24G" ;;
        step-audio-editx) printf '%s\n' "32G" ;;
        breeze-tts2) printf '%s\n' "32G" ;;
        cosyvoice3) printf '%s\n' "32G" ;;
        higgs-audio-v3) printf '%s\n' "48G" ;;
        inworld-tts-poc) printf '%s\n' "48G" ;;
        *) printf '%s\n' "" ;;
    esac
}

MODE="smoke"
CLUSTER="auto"
ACCOUNT="${CSC_PROJECT:-}"
PARTITION_OVERRIDE=""
TIME_OVERRIDE=""
CPUS_OVERRIDE=""
GRES_OVERRIDE=""
MEM_OVERRIDE=""
RUNNER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION_OVERRIDE="$2"
            shift 2
            ;;
        --time)
            TIME_OVERRIDE="$2"
            shift 2
            ;;
        --cpus)
            CPUS_OVERRIDE="$2"
            shift 2
            ;;
        --gres)
            GRES_OVERRIDE="$2"
            shift 2
            ;;
        --mem)
            MEM_OVERRIDE="$2"
            shift 2
            ;;
        --)
            shift
            RUNNER_ARGS+=("$@")
            break
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ -n "$ACCOUNT" ]] || {
    echo "set CSC_PROJECT or pass --account <project>" >&2
    exit 2
}

if [[ "$CLUSTER" == "auto" ]]; then
    CLUSTER="$(detect_cluster_name)"
fi

case "$CLUSTER" in
    mahti|puhti|roihu)
        ;;
    local)
        CLUSTER="mahti"
        ;;
    *)
        echo "unknown cluster: $CLUSTER" >&2
        echo "expected auto, mahti, puhti, or roihu" >&2
        exit 2
        ;;
esac

case "$MODE" in
    smoke)
        TIME="00:15:00"
        case "$CLUSTER" in
            puhti)
                PARTITION="gputest"
                CPUS="10"
                GRES="gpu:v100:1"
                ;;
            roihu)
                PARTITION="gputest"
                CPUS="32"
                GRES="gpu:gh200:1"
                ;;
            *)
                PARTITION="gputest"
                CPUS="16"
                GRES="gpu:a100:1,nvme:900"
                ;;
        esac
        ;;
    full)
        TIME="$(resolve_runtime "$MODEL_ID")"
        case "$CLUSTER" in
            puhti)
                PARTITION="gpu"
                CPUS="10"
                GRES="gpu:v100:1"
                ;;
            roihu)
                PARTITION="gpumedium"
                CPUS="32"
                GRES="gpu:gh200:1"
                ;;
            *)
                PARTITION="gpusmall"
                CPUS="32"
                GRES="gpu:a100:1,nvme:900"
                ;;
        esac
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "expected smoke or full" >&2
        exit 2
        ;;
esac

PARTITION="${PARTITION_OVERRIDE:-$PARTITION}"
TIME="${TIME_OVERRIDE:-$TIME}"
CPUS="${CPUS_OVERRIDE:-$CPUS}"
GRES="${GRES_OVERRIDE:-$GRES}"

SLURM_LOG_DIR="$REPO_ROOT/experiments/tts_naturalness/generated/runs/slurm"
mkdir -p "$SLURM_LOG_DIR"
_RESOLVED_MEM="$(resolve_mem "$MODEL_ID")"
MEM="${MEM_OVERRIDE:-$_RESOLVED_MEM}"
SBATCH_MEM_ARGS=()
if [[ -n "$MEM" ]]; then
    SBATCH_MEM_ARGS=(--mem "$MEM")
fi

JOB_NAME="tts-${CLUSTER}-${MODEL_ID}"

printf 'submitting %s on %s (%s, %s, %s, %s)\n' "$MODEL_ID" "$CLUSTER" "$PARTITION" "$TIME" "$CPUS" "$GRES" >&2

exec sbatch \
    --account "$ACCOUNT" \
    --job-name "$JOB_NAME" \
    --partition "$PARTITION" \
    --time "$TIME" \
    --ntasks 1 \
    --cpus-per-task "$CPUS" \
    --gres "$GRES" \
    "${SBATCH_MEM_ARGS[@]}" \
    --output "$SLURM_LOG_DIR/%x-%j.out" \
    --error "$SLURM_LOG_DIR/%x-%j.out" \
    --chdir "$REPO_ROOT" \
    --export "ALL,CSC_CLUSTER_NAME=$CLUSTER,CSC_PROJECT=$ACCOUNT" \
    "$SCRIPT_DIR/run_model.slurm" \
    "$MODEL_ID" \
    "${RUNNER_ARGS[@]}"
