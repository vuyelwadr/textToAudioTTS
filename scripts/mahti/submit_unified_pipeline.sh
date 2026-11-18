#!/usr/bin/env bash
#
# Thin wrapper around scripts/mahti/submit_audiobook_video.sh that targets the
# unified pipeline entrypoint (scripts/audiobook_video_pipeline.py) instead of the
# active-line video runner. It does not implement any cluster/partition/CPU/GPU
# resource logic of its own -- that would duplicate scripts/mahti/submit_audiobook_video.sh
# (which itself wraps scripts/mahti/common.sh's cluster detection). This script only
# knows how to route "--input-type/--input" plus the shared resource flags through to
# that existing, fully dynamic submitter via its --runner and --gpus flags.
#
# This is the "anyone can paste a YouTube URL (or a local text/audio file) and queue
# the job" entrypoint: one command, any cluster, any input type, no hand-written
# per-book .slurm file.
#
# usage: scripts/mahti/submit_unified_pipeline.sh --input-type {youtube,text,audio} --input <url|path>
#        [--cluster auto|mahti|puhti|roihu] [--mode smoke|full] [--resource-mode gpu|cpu]
#        [--cpus N] [--gpus N] [--account <project>] [-- <extra audiobook_video_pipeline.py args>]
#
# Examples:
#   scripts/mahti/submit_unified_pipeline.sh \
#     --input-type youtube --input https://www.youtube.com/watch?v=emn3Hq977BU \
#     --cluster auto --mode smoke --resource-mode cpu --account project_2010549 \
#     -- --slug aTalkWithAWayfarerVerify --use-youtube-chapters --embed-chapters
#
#   scripts/mahti/submit_unified_pipeline.sh \
#     --input-type text --input library/tolstoy/aSparkNeglectedBurnsTheHouse/aSparkNeglectedBurnsTheHouseTranscript.json \
#     --cluster mahti --mode full --resource-mode gpu --gpus 1 --account project_2010549
#
#   scripts/mahti/submit_unified_pipeline.sh \
#     --input-type audio --input /scratch/project_2010549/$USER/some_narration.wav \
#     --cluster roihu --mode full --resource-mode gpu --gpus 2 --account project_2010549 \
#     -- --manual-chapters /scratch/project_2010549/$USER/some_narration.chapters.json

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: scripts/mahti/submit_unified_pipeline.sh --input-type {youtube,text,audio} --input <url|path> [--cluster auto|mahti|puhti|roihu] [--mode smoke|full] [--resource-mode gpu|cpu] [--cpus N] [--gpus N] [--account <project>] [-- <extra audiobook_video_pipeline.py args>]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

INPUT_TYPE=""
INPUT=""
CLUSTER="auto"
MODE="full"
RESOURCE_MODE="gpu"
ACCOUNT="${CSC_PROJECT:-}"
CPUS_OVERRIDE=""
GPUS_OVERRIDE=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-type)
            INPUT_TYPE="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
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
        --cpus)
            CPUS_OVERRIDE="$2"
            shift 2
            ;;
        --gpus)
            GPUS_OVERRIDE="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --)
            shift
            PASSTHROUGH_ARGS+=("$@")
            break
            ;;
        *)
            # Forward-compatible with any audiobook_video_pipeline.py flag this
            # wrapper does not know about by name (--slug, --output-root,
            # --paragraph-strategy, --use-youtube-chapters, --manual-chapters,
            # --transcript-source, --embed-chapters, --force-*, --dry-run, ...),
            # mirroring submit_audiobook_video.sh's own permissive passthrough.
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ -n "$INPUT_TYPE" ]] || {
    echo "--input-type {youtube,text,audio} is required" >&2
    exit 2
}

case "$INPUT_TYPE" in
    youtube|text|audio)
        ;;
    *)
        echo "unknown --input-type: $INPUT_TYPE" >&2
        echo "expected youtube, text, or audio" >&2
        exit 2
        ;;
esac

[[ -n "$INPUT" ]] || {
    echo "--input <url-or-path> is required" >&2
    exit 2
}

# --cluster/--mode/--resource-mode/--account validity, and the CPU/GPU resource
# table itself (including --gpus N -> --gres synthesis), are all owned by
# submit_audiobook_video.sh -- deliberately not re-checked or re-implemented here.
SUBMIT_ARGS=(
    --cluster "$CLUSTER"
    --mode "$MODE"
    --resource-mode "$RESOURCE_MODE"
    --account "$ACCOUNT"
    --runner "$REPO_ROOT/scripts/audiobook_video_pipeline.py"
)

if [[ -n "$CPUS_OVERRIDE" ]]; then
    SUBMIT_ARGS+=(--cpus "$CPUS_OVERRIDE")
fi
if [[ -n "$GPUS_OVERRIDE" ]]; then
    SUBMIT_ARGS+=(--gpus "$GPUS_OVERRIDE")
fi

exec bash "$SCRIPT_DIR/submit_audiobook_video.sh" \
    "${SUBMIT_ARGS[@]}" \
    -- \
    --input-type "$INPUT_TYPE" \
    --input "$INPUT" \
    --submit-target-is-cluster-job \
    "${PASSTHROUGH_ARGS[@]}"
