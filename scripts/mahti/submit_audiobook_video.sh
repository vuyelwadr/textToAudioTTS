#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/mahti/submit_audiobook_video.sh [--cluster auto|mahti|puhti|roihu] [--mode smoke|full] [--resource-mode gpu|cpu] [--account <project>] [submit options...] [-- runner args...]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

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

MODE="full"
CLUSTER="auto"
RESOURCE_MODE="gpu"
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
        --resource-mode)
            RESOURCE_MODE="$2"
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

if [[ ${#RUNNER_ARGS[@]} -eq 0 ]]; then
    echo "pass video runner arguments such as --summary ... --model ... --show-active-line" >&2
    exit 2
fi

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
        case "$RESOURCE_MODE" in
            cpu)
                TIME="00:45:00"
                case "$CLUSTER" in
                    puhti)
                        PARTITION="small"
                        CPUS="10"
                        ;;
                    roihu)
                        PARTITION="small"
                        CPUS="16"
                        ;;
                    *)
                        PARTITION="small"
                        CPUS="8"
                        ;;
                esac
                GRES=""
                ;;
            gpu)
                TIME="00:20:00"
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
            *)
                echo "unknown resource mode: $RESOURCE_MODE" >&2
                echo "expected gpu or cpu" >&2
                exit 2
                ;;
        esac
        ;;
    full)
        case "$RESOURCE_MODE" in
            cpu)
                TIME="08:00:00"
                case "$CLUSTER" in
                    puhti)
                        PARTITION="small"
                        CPUS="20"
                        ;;
                    roihu)
                        PARTITION="small"
                        CPUS="32"
                        ;;
                    *)
                        PARTITION="small"
                        CPUS="32"
                        ;;
                esac
                GRES=""
                ;;
            gpu)
                TIME="04:00:00"
                case "$CLUSTER" in
                    puhti)
                        PARTITION="gpu"
                        CPUS="20"
                        GRES="gpu:v100:2"
                        ;;
                    roihu)
                        PARTITION="gpumedium"
                        CPUS="32"
                        GRES="gpu:gh200:2"
                        ;;
                    *)
                        PARTITION="gpusmall"
                        CPUS="32"
                        GRES="gpu:a100:2,nvme:900"
                        ;;
                esac
                ;;
            *)
                echo "unknown resource mode: $RESOURCE_MODE" >&2
                echo "expected gpu or cpu" >&2
                exit 2
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
MEM="${MEM_OVERRIDE:-}"

SLURM_LOG_DIR="$REPO_ROOT/experiments/tts_naturalness/generated/runs/slurm"
mkdir -p "$SLURM_LOG_DIR"

SBATCH_MEM_ARGS=()
if [[ -n "$MEM" ]]; then
    SBATCH_MEM_ARGS=(--mem "$MEM")
fi

JOB_NAME="tts-video-${CLUSTER}-${RESOURCE_MODE}"

printf 'submitting audiobook video on %s (%s, %s, %s, %s)\n' \
    "$CLUSTER" "$PARTITION" "$TIME" "$CPUS" "${GRES:-cpu-only}" >&2

SBATCH_CMD=(
    sbatch
    --account "$ACCOUNT"
    --job-name "$JOB_NAME"
    --partition "$PARTITION"
    --time "$TIME"
    --ntasks 1
    --cpus-per-task "$CPUS"
)

if [[ -n "$GRES" ]]; then
    SBATCH_CMD+=(--gres "$GRES")
fi

if [[ ${#SBATCH_MEM_ARGS[@]} -gt 0 ]]; then
    SBATCH_CMD+=("${SBATCH_MEM_ARGS[@]}")
fi

SBATCH_CMD+=(
    --output "$SLURM_LOG_DIR/%x-%j.out"
    --error "$SLURM_LOG_DIR/%x-%j.out"
    --chdir "$REPO_ROOT"
    --export "ALL,CSC_CLUSTER_NAME=$CLUSTER,CSC_PROJECT=$ACCOUNT"
    "$SCRIPT_DIR/run_audiobook_video.sh"
    "${RUNNER_ARGS[@]}"
)

exec "${SBATCH_CMD[@]}"
