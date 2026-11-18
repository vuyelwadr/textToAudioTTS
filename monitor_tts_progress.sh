#!/bin/bash
# TTS Progress Monitor
# Usage: ./monitor_tts_progress.sh [interval_in_seconds]
# Default interval: 30 seconds
# Set REMOTE_CLUSTERS=puhti,mahti to compare both clusters in one view.

REMOTE_USER="${REMOTE_USER:-ruwododa}"
REMOTE_CLUSTER="${REMOTE_CLUSTER:-}"
REMOTE_CLUSTERS="${REMOTE_CLUSTERS:-}"
if [ -z "$REMOTE_CLUSTERS" ]; then
    if [ -n "$REMOTE_CLUSTER" ]; then
        REMOTE_CLUSTERS="$REMOTE_CLUSTER"
    else
        REMOTE_CLUSTERS="puhti,mahti"
    fi
fi
case "$REMOTE_CLUSTERS" in
    both|all)
        REMOTE_CLUSTERS="puhti,mahti"
        ;;
esac
REMOTE_CLUSTER_LIST=()
OLD_IFS="$IFS"
IFS=,
for cluster in $REMOTE_CLUSTERS; do
    cluster="${cluster// /}"
    [ -n "$cluster" ] && REMOTE_CLUSTER_LIST+=("$cluster")
done
IFS="$OLD_IFS"
if [ "${#REMOTE_CLUSTER_LIST[@]}" -eq 0 ]; then
    echo "No clusters selected; set REMOTE_CLUSTER or REMOTE_CLUSTERS" >&2
    exit 1
fi
INTERVAL="${1:-30}"
MAX_DISCOVERED_ROOTS="${MAX_DISCOVERED_ROOTS:-5}"
MAX_DISCOVERY_DEPTH="${MAX_DISCOVERY_DEPTH:-8}"

REMOTE_HOST=""
REMOTE_BASE=""
REMOTE_CLUSTER_LABEL=""

PUHTI_COMPLETED_TOTAL=0
PUHTI_RUNTIME_SECONDS=0
PUHTI_MAX_ETA_SECONDS=0
MAHTI_COMPLETED_TOTAL=0
MAHTI_RUNTIME_SECONDS=0
MAHTI_MAX_ETA_SECONDS=0

set_cluster_context() {
    case "$1" in
        puhti)
            REMOTE_HOST="${PUHTI_HOST:-puhti.csc.fi}"
            REMOTE_BASE="${PUHTI_BASE:-/scratch/project_2010549/ruwododa/textToAudioTTS-puhti}"
            REMOTE_CLUSTER_LABEL="puhti"
            ;;
        mahti)
            REMOTE_HOST="${MAHTI_HOST:-mahti.csc.fi}"
            REMOTE_BASE="${MAHTI_BASE:-/scratch/project_2010549/ruwododa/textToAudioTTS-mahti}"
            REMOTE_CLUSTER_LABEL="mahti"
            ;;
        *)
            echo "Unknown cluster: $1" >&2
            exit 1
            ;;
    esac
}

reset_cluster_totals() {
    case "$1" in
        puhti)
            PUHTI_COMPLETED_TOTAL=0
            PUHTI_RUNTIME_SECONDS=0
            PUHTI_MAX_ETA_SECONDS=0
            ;;
        mahti)
            MAHTI_COMPLETED_TOTAL=0
            MAHTI_RUNTIME_SECONDS=0
            MAHTI_MAX_ETA_SECONDS=0
            ;;
    esac
}

count_running_jobs() {
    printf '%s\n' "$1" | awk -F'|' '
        $3 ~ /^(RUNNING|COMPLETING|CONFIGURING|R|CG|CF)$/ { count++ }
        END { print count + 0 }
    '
}

query_cluster_jobs_parallel() {
    local query_dir="$1"
    local cluster
    local outfile
    local pids=()

    for cluster in "${REMOTE_CLUSTER_LIST[@]}"; do
        outfile="$query_dir/$cluster.jobs"
        (
            set_cluster_context "$cluster"
            remote_jobs > "$outfile"
        ) &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
}

select_display_clusters() {
    local query_dir="$1"
    local cluster
    local jobs_output
    local running_jobs

    DISPLAY_CLUSTER_LIST=()
    ACTIVE_CLUSTER_LIST=()

    for cluster in "${REMOTE_CLUSTER_LIST[@]}"; do
        jobs_output=""
        if [ -f "$query_dir/$cluster.jobs" ]; then
            jobs_output="$(cat "$query_dir/$cluster.jobs")"
        fi
        running_jobs="$(count_running_jobs "$jobs_output")"
        if [ "$running_jobs" -gt 0 ]; then
            ACTIVE_CLUSTER_LIST+=("$cluster")
        fi
    done

    if [ "${#ACTIVE_CLUSTER_LIST[@]}" -gt 0 ]; then
        DISPLAY_CLUSTER_LIST=("${ACTIVE_CLUSTER_LIST[@]}")
        DISPLAY_MODE="running jobs found; showing active cluster(s)"
    else
        DISPLAY_CLUSTER_LIST=("${REMOTE_CLUSTER_LIST[@]}")
        DISPLAY_MODE="no running jobs found; showing all selected cluster(s)"
    fi
}

SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout=12
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
)

remote_jobs() {
    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" bash -s -- "$REMOTE_USER" <<'REMOTE'
user="$1"
squeue -h -u "$user" -o '%i|%j|%T|%M|%N' 2>/dev/null || true
REMOTE
}

remote_job_detail() {
    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" bash -s -- "$1" <<'REMOTE'
jobid="$1"
scontrol show job -o "$jobid" 2>/dev/null || true
REMOTE
}

discover_run_roots() {
    local search_base="$1"
    local limit="${2:-$MAX_DISCOVERED_ROOTS}"

    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" bash -s -- "$search_base" "$limit" "$MAX_DISCOVERY_DEPTH" <<'REMOTE'
search_base="$1"
limit="$2"
depth="$3"

[ -d "$search_base" ] || exit 0

find "$search_base" -maxdepth "$depth" -type f 2>/dev/null |
grep -E '/(audiobook_summary\.json|render_manifest\.json|[^/]+_manifest\.json|chunks/[^/]+\.wav)$' |
while IFS= read -r path; do
    case "$path" in
        */audiobook_summary.json)
            root=$(dirname "$path")
            source_kind="summary"
            ;;
        */render_manifest.json)
            root=$(dirname "$(dirname "$path")")
            source_kind="render_manifest"
            ;;
        */chunks/*_manifest.json)
            root=$(dirname "$(dirname "$path")")
            source_kind="chunk_manifest"
            ;;
        */chunks/*.wav)
            root=$(dirname "$(dirname "$(dirname "$path")")")
            source_kind="chunk_output"
            ;;
        *)
            continue
            ;;
    esac

    mtime=$(stat -c %Y "$path" 2>/dev/null || echo 0)
    printf '%s|%s|%s|%s\n' "$mtime" "$root" "$source_kind" "$path"
done |
    sort -nr |
    awk -F'|' '!seen[$2]++ { print $2 "|" $3 "|" $4 "|" $1 }' |
    head -n "$limit"
REMOTE
}

inspect_run_root() {
    local root="$1"
    local elapsed_seconds="$2"

    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" python3 - "$root" "$elapsed_seconds" <<'REMOTE'
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
elapsed_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 0
now = time.time()
job_start_epoch = now - elapsed_seconds if elapsed_seconds > 0 else None


def emit(*parts):
    print("|".join(str(part) for part in parts))


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_from_manifest(data):
    if isinstance(data, dict):
        for key in ("render_chunk_count", "chunk_count", "total_chunks"):
            value = data.get(key)
            if isinstance(value, int):
                return value
        chunks = data.get("chunks")
        if isinstance(chunks, list):
            return len(chunks)
        if isinstance(chunks, dict):
            return len(chunks)
    if isinstance(data, list):
        return len(data)
    return None


def human_duration(seconds):
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_rate(completed, elapsed):
    if completed <= 0 or elapsed <= 0:
        return "n/a"
    return f"{completed * 3600 / elapsed:.2f} chunks/hr"


def format_avg_time(completed, elapsed):
    if completed <= 0 or elapsed <= 0:
        return "n/a"
    return f"{elapsed / completed:.1f} sec/chunk"


def latest_chunk_age(chunks_dir: Path):
    if not chunks_dir.exists():
        return "n/a"
    wavs = list(chunks_dir.glob("*.wav"))
    if not wavs:
        return "n/a"
    latest = max(wavs, key=lambda path: path.stat().st_mtime)
    return human_duration(now - latest.stat().st_mtime)


if not root.exists():
    sys.exit(0)

summary_path = root / "audiobook_summary.json"
chunk_manifest_path = next(iter(sorted((root / "chunks").glob("*_manifest.json"))), None) if (root / "chunks").exists() else None
summary_data = load_json(summary_path) if summary_path.exists() else None
chunk_manifest_data = load_json(chunk_manifest_path) if chunk_manifest_path and chunk_manifest_path.exists() else None

summary_chunk_count = None
summary_render_count = None
summary_output_dir = ""
summary_work_dir = ""
summary_models = []

if isinstance(summary_data, dict):
    summary_chunk_count = summary_data.get("chunk_count") if isinstance(summary_data.get("chunk_count"), int) else None
    summary_render_count = summary_data.get("render_chunk_count") if isinstance(summary_data.get("render_chunk_count"), int) else None
    summary_output_dir = summary_data.get("output_dir") or ""
    summary_work_dir = summary_data.get("work_dir") or ""
    models = summary_data.get("models")
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict):
                name = model.get("model") or model.get("label") or model.get("name")
                if name:
                    summary_models.append(str(name))
            elif model:
                summary_models.append(str(model))

chunk_manifest_count = count_from_manifest(chunk_manifest_data)

emit("ROOT", str(root))
if summary_path.exists():
    emit(
        "SUMMARY",
        str(summary_path),
        summary_chunk_count if summary_chunk_count is not None else "",
        summary_render_count if summary_render_count is not None else "",
        str(chunk_manifest_path) if chunk_manifest_path else "",
        summary_output_dir,
        summary_work_dir,
        ",".join(summary_models),
    )

if chunk_manifest_path and chunk_manifest_path.exists():
    emit(
        "CHUNK_MANIFEST",
        str(chunk_manifest_path),
        chunk_manifest_count if chunk_manifest_count is not None else "",
    )

model_dirs = []
seen_dirs = set()

for child in sorted(root.iterdir(), key=lambda path: path.name):
    if not child.is_dir() or child.name in {"chunks", "logs"}:
        continue
    if (child / "chunks").exists() or (child / "render_manifest.json").exists():
        seen_dirs.add(child)
        model_dirs.append(child)

for model_name in summary_models:
    candidate = root / model_name
    if candidate not in seen_dirs:
        seen_dirs.add(candidate)
        model_dirs.append(candidate)

for model_dir in sorted(model_dirs, key=lambda path: path.name):
    chunks_dir = model_dir / "chunks"
    render_manifest_path = model_dir / "render_manifest.json"
    render_data = load_json(render_manifest_path) if render_manifest_path.exists() else None

    chunk_outputs = list(chunks_dir.glob("*.wav")) if chunks_dir.exists() else []
    completed = len(chunk_outputs)
    current_job_completed = (
        sum(1 for chunk_output in chunk_outputs if chunk_output.stat().st_mtime >= job_start_epoch)
        if job_start_epoch is not None
        else 0
    )
    latest_age = latest_chunk_age(chunks_dir)
    render_count = count_from_manifest(render_data)
    render_status = ""
    final_output = ""
    if isinstance(render_data, dict):
        render_status = str(render_data.get("status") or "")
        final_output = str(render_data.get("final_output") or "")

    total = render_count
    if total is None and summary_render_count is not None:
        total = summary_render_count
    if total is None and summary_chunk_count is not None:
        total = summary_chunk_count
    if total is None and chunk_manifest_count is not None:
        total = chunk_manifest_count

    if total is not None and total > 0:
        pct = f"{(completed * 100 / total):.1f}"
    else:
        pct = "n/a"

    if total is not None and total <= completed:
        eta = "00:00"
    elif elapsed_seconds > 0 and current_job_completed > 0 and total is not None and total > completed:
        eta = human_duration((total - completed) * elapsed_seconds / current_job_completed)
    else:
        eta = "n/a"

    if elapsed_seconds > 0 and current_job_completed > 0:
        speed = format_rate(current_job_completed, elapsed_seconds)
        avg = format_avg_time(current_job_completed, elapsed_seconds)
    else:
        speed = "n/a"
        avg = "n/a"

    emit(
        "MODEL",
        model_dir.name,
        str(model_dir),
        completed,
        total if total is not None else "",
        pct,
        current_job_completed,
        speed,
        avg,
        eta,
        latest_age,
        str(render_manifest_path) if render_manifest_path.exists() else "",
        render_count if render_count is not None else "",
        render_status,
        final_output,
    )
REMOTE
}

remote_gpu_usage() {
    ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$REMOTE_HOST" bash -s -- "$1" <<'REMOTE'
jobid="$1"

if ! command -v srun >/dev/null 2>&1; then
    exit 0
fi

cmd="srun --jobid=\"$jobid\" --nodes=1 --ntasks=1 --exact --overlap --quiet bash -lc 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null | awk -F', *' 'NF >= 6 { printf \"GPU|%s|%s|%s|%s|%s|%s\\n\", \$1, \$2, \$3, \$4, \$5, \$6 }'"

if command -v timeout >/dev/null 2>&1; then
    timeout 20s bash -lc "$cmd"
else
    bash -lc "$cmd"
fi
REMOTE
}

parse_slurm_field() {
    local text="$1"
    local key="$2"

    printf '%s\n' "$text" | awk -v key="$key" '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ "^" key "=") {
                    sub("^" key "=", "", $i)
                    print $i
                    exit
                }
            }
        }
    '
}

elapsed_to_seconds() {
    local value="$1"
    local days=0
    local time_part="$value"
    local hours=0
    local minutes=0
    local seconds=0

    case "$value" in
        ""|n/a|N/A)
            echo 0
            return
            ;;
    esac

    if [ -z "$value" ]; then
        echo 0
        return
    fi

    case "$value" in
        *-*)
            days="${value%%-*}"
            time_part="${value#*-}"
            ;;
    esac

    IFS=: read -r hours minutes seconds <<EOF
$time_part
EOF

    if [ -z "$seconds" ]; then
        seconds="$minutes"
        minutes="$hours"
        hours=0
    fi

    printf '%s\n' "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

duration_to_seconds() {
    local value="$1"

    case "$value" in
        ""|n/a|N/A)
            echo 0
            return
            ;;
    esac

    elapsed_to_seconds "$value"
}

format_cluster_avg_seconds() {
    local elapsed_seconds="$1"
    local completed_chunks="$2"

    awk -v elapsed="$elapsed_seconds" -v completed="$completed_chunks" 'BEGIN {
        if (elapsed <= 0 || completed <= 0) {
            print "n/a"
        } else {
            printf "%.1f", elapsed / completed
        }
    }'
}

format_cluster_chunks_per_hour() {
    local elapsed_seconds="$1"
    local completed_chunks="$2"

    awk -v elapsed="$elapsed_seconds" -v completed="$completed_chunks" 'BEGIN {
        if (elapsed <= 0 || completed <= 0) {
            print "n/a"
        } else {
            printf "%.2f", (completed * 3600) / elapsed
        }
    }'
}

record_cluster_totals() {
    local cluster="$1"
    local completed="$2"
    local elapsed_text="$3"
    local eta_text="$4"
    local elapsed_seconds
    local eta_seconds

    case "$completed" in
        ""|*[!0-9]*)
            completed=0
            ;;
    esac

    elapsed_seconds="$(elapsed_to_seconds "$elapsed_text")"
    eta_seconds="$(duration_to_seconds "$eta_text")"

    case "$cluster" in
        puhti)
            if [ "$elapsed_seconds" -gt "$PUHTI_RUNTIME_SECONDS" ]; then
                PUHTI_RUNTIME_SECONDS="$elapsed_seconds"
            fi
            if [ "$completed" -gt 0 ] && [ "$elapsed_seconds" -gt 0 ]; then
                PUHTI_COMPLETED_TOTAL=$((PUHTI_COMPLETED_TOTAL + completed))
            fi
            if [ "$eta_seconds" -gt "$PUHTI_MAX_ETA_SECONDS" ]; then
                PUHTI_MAX_ETA_SECONDS="$eta_seconds"
            fi
            ;;
        mahti)
            if [ "$elapsed_seconds" -gt "$MAHTI_RUNTIME_SECONDS" ]; then
                MAHTI_RUNTIME_SECONDS="$elapsed_seconds"
            fi
            if [ "$completed" -gt 0 ] && [ "$elapsed_seconds" -gt 0 ]; then
                MAHTI_COMPLETED_TOTAL=$((MAHTI_COMPLETED_TOTAL + completed))
            fi
            if [ "$eta_seconds" -gt "$MAHTI_MAX_ETA_SECONDS" ]; then
                MAHTI_MAX_ETA_SECONDS="$eta_seconds"
            fi
            ;;
    esac
}

print_cluster_summary() {
    local cluster="$1"
    local completed_total
    local runtime_seconds
    local max_eta_seconds
    local avg_seconds
    local chunks_per_hour
    local eta_text

    case "$cluster" in
        puhti)
            completed_total="$PUHTI_COMPLETED_TOTAL"
            runtime_seconds="$PUHTI_RUNTIME_SECONDS"
            max_eta_seconds="$PUHTI_MAX_ETA_SECONDS"
            ;;
        mahti)
            completed_total="$MAHTI_COMPLETED_TOTAL"
            runtime_seconds="$MAHTI_RUNTIME_SECONDS"
            max_eta_seconds="$MAHTI_MAX_ETA_SECONDS"
            ;;
        *)
            return
            ;;
    esac

    avg_seconds="$(format_cluster_avg_seconds "$runtime_seconds" "$completed_total")"
    chunks_per_hour="$(format_cluster_chunks_per_hour "$runtime_seconds" "$completed_total")"
    if [ "$max_eta_seconds" -gt 0 ]; then
        eta_text="$(format_duration "$max_eta_seconds")"
    else
        eta_text="n/a"
    fi

    printf "Cluster summary for %s:\n" "$cluster"
    printf "  current-job chunks: %s\n" "$completed_total"
    printf "  avg: %s sec/chunk | speed: %s chunks/hr | rough ETA: %s\n" "$avg_seconds" "$chunks_per_hour" "$eta_text"
    printf "\n"
}

print_cluster_comparison() {
    local puhti_avg
    local mahti_avg
    local faster_cluster=""
    local slower_cluster=""
    local faster_avg=""
    local slower_avg=""
    local gap=""
    local pct=""

    puhti_avg="$(format_cluster_avg_seconds "$PUHTI_RUNTIME_SECONDS" "$PUHTI_COMPLETED_TOTAL")"
    mahti_avg="$(format_cluster_avg_seconds "$MAHTI_RUNTIME_SECONDS" "$MAHTI_COMPLETED_TOTAL")"

    if [ "$puhti_avg" = "n/a" ] || [ "$mahti_avg" = "n/a" ]; then
        printf "Cluster comparison: n/a (need active data from both clusters)\n\n"
        return
    fi

    if awk -v puhti="$puhti_avg" -v mahti="$mahti_avg" 'BEGIN { exit !(mahti < puhti) }'; then
        faster_cluster="mahti"
        slower_cluster="puhti"
        faster_avg="$mahti_avg"
        slower_avg="$puhti_avg"
    else
        faster_cluster="puhti"
        slower_cluster="mahti"
        faster_avg="$puhti_avg"
        slower_avg="$mahti_avg"
    fi

    gap="$(awk -v fast="$faster_avg" -v slow="$slower_avg" 'BEGIN { printf "%.1f", slow - fast }')"
    pct="$(awk -v gap="$gap" -v slow="$slower_avg" 'BEGIN { if (slow <= 0) print "n/a"; else printf "%.1f", (gap / slow) * 100 }')"

    printf "Cluster comparison:\n"
    printf "  faster: %s by %s sec/chunk (%s%%)\n" "$faster_cluster" "$gap" "$pct"
    printf "  slower: %s by %s sec/chunk\n" "$slower_cluster" "$gap"
    printf "  puhti avg: %s sec/chunk | mahti avg: %s sec/chunk\n" "$puhti_avg" "$mahti_avg"
    printf "\n"
}

format_duration() {
    local total="$1"
    local days hours minutes seconds

    if [ -z "$total" ] || [ "$total" -le 0 ] 2>/dev/null; then
        printf 'n/a\n'
        return
    fi

    days=$((total / 86400))
    total=$((total % 86400))
    hours=$((total / 3600))
    total=$((total % 3600))
    minutes=$((total / 60))
    seconds=$((total % 60))

    if [ "$days" -gt 0 ]; then
        printf '%dd %02d:%02d:%02d\n' "$days" "$hours" "$minutes" "$seconds"
    elif [ "$hours" -gt 0 ]; then
        printf '%02d:%02d:%02d\n' "$hours" "$minutes" "$seconds"
    else
        printf '%02d:%02d\n' "$minutes" "$seconds"
    fi
}

is_running_state() {
    case "$1" in
        RUNNING|COMPLETING|CONFIGURING|R|CG|CF)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

clear_screen() {
    if [ -t 1 ]; then
        clear
    fi
}

root_seen_contains() {
    local needle="$1"
    case "
$ROOT_SEEN
" in
        *"
$needle
"*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

root_seen_add() {
    local value="$1"

    if [ -z "$ROOT_SEEN" ]; then
        ROOT_SEEN="$value"
    else
        ROOT_SEEN="${ROOT_SEEN}
$value"
    fi
}

print_job_table() {
    local jobs_output="$1"
    local job_count=0

    printf "Active %s jobs:\n" "$REMOTE_CLUSTER_LABEL"
    if [ -z "$jobs_output" ]; then
        printf "  (none)\n\n"
        return
    fi

    printf "  %-12s %-22s %-12s %-12s %-24s\n" "JOB ID" "NAME" "STATE" "ELAPSED" "NODE"
    printf "  %-12s %-22s %-12s %-12s %-24s\n" "------------" "----------------------" "------------" "------------" "------------------------"

    while IFS='|' read -r jobid job_name state elapsed node; do
        [ -n "$jobid" ] || continue
        [ -n "$node" ] || node="-"
        printf "  %-12s %-22.22s %-12.12s %-12s %-24.24s\n" "$jobid" "$job_name" "$state" "$elapsed" "$node"
        job_count=$((job_count + 1))
    done <<EOF
$jobs_output
EOF

    [ "$job_count" -gt 0 ] && printf "\n"
}

print_gpu_section() {
    local jobid="$1"
    local job_name="$2"
    local state="$3"
    local job_detail="$4"
    local gpu_output
    local alloc

    if ! is_running_state "$state"; then
        return
    fi

    gpu_output="$(remote_gpu_usage "$jobid" 2>/dev/null)"
    if [ -n "$gpu_output" ]; then
        printf "  GPU usage for job %s (%s):\n" "$jobid" "$job_name"
        while IFS='|' read -r tag gpu_index gpu_name util_gpu util_mem mem_used mem_total; do
            [ "$tag" = "GPU" ] || continue
            printf "    GPU %s %s | util %s%% / mem util %s%% | mem %s/%s MiB\n" \
                "$gpu_index" "$gpu_name" "$util_gpu" "$util_mem" "$mem_used" "$mem_total"
        done <<EOF
$gpu_output
EOF
    else
        alloc="$(parse_slurm_field "$job_detail" "Gres")"
        [ -z "$alloc" ] && alloc="$(parse_slurm_field "$job_detail" "TRES")"
        if [ -n "$alloc" ]; then
            printf "  GPU usage for job %s (%s): unavailable (allocation: %s)\n" "$jobid" "$job_name" "$alloc"
        else
            printf "  GPU usage for job %s (%s): unavailable\n" "$jobid" "$job_name"
        fi
    fi
}

print_root_report() {
    local root="$1"
    local jobid="$2"
    local job_name="$3"
    local state="$4"
    local elapsed="$5"
    local node="$6"
    local source_kind="$7"
    local source_path="$8"
    local cluster="$9"
    local report="${10}"
    local runtime_basis
    local saw_model=0
    local model_name
    local model_path
    local completed
    local total
    local pct
    local speed
    local avg
    local eta
    local last_chunk_age
    local render_manifest
    local render_count
    local render_status
    local final_output
    local summary_path
    local summary_chunk_count
    local summary_render_count
    local chunk_manifest_path
    local chunk_manifest_count
    local summary_output_dir
    local summary_work_dir
    local summary_models

    runtime_basis="$(format_duration "$(elapsed_to_seconds "$elapsed")")"

    printf "Run tree: %s\n" "$root"
    printf "  related job: %s | %s | %s | %s | %s\n" "$jobid" "$job_name" "$state" "$elapsed" "$node"
    printf "  discovered via: %s (%s)\n" "$source_kind" "$source_path"
    printf "  runtime basis: %s\n" "$runtime_basis"
    printf "  cluster: %s\n" "$cluster"

    while IFS='|' read -r tag a b c d e f g h i j k l m n; do
        case "$tag" in
            ROOT)
                :
                ;;
            SUMMARY)
                summary_path="$a"
                summary_chunk_count="$b"
                summary_render_count="$c"
                chunk_manifest_path="$d"
                summary_output_dir="$e"
                summary_work_dir="$f"
                summary_models="$g"
                printf "  summary manifest: %s\n" "$summary_path"
                [ -n "$summary_chunk_count" ] && printf "    chunk count: %s\n" "$summary_chunk_count"
                [ -n "$summary_render_count" ] && printf "    render target: %s\n" "$summary_render_count"
                [ -n "$chunk_manifest_path" ] && printf "    chunk manifest: %s\n" "$chunk_manifest_path"
                [ -n "$summary_output_dir" ] && printf "    output dir: %s\n" "$summary_output_dir"
                [ -n "$summary_work_dir" ] && printf "    work dir: %s\n" "$summary_work_dir"
                [ -n "$summary_models" ] && printf "    models: %s\n" "$summary_models"
                ;;
            CHUNK_MANIFEST)
                chunk_manifest_path="$a"
                chunk_manifest_count="$b"
                printf "  input manifest: %s\n" "$chunk_manifest_path"
                [ -n "$chunk_manifest_count" ] && printf "    chunks: %s\n" "$chunk_manifest_count"
                ;;
            MODEL)
                saw_model=1
                model_name="$a"
                model_path="$b"
                completed="$c"
                total="$d"
                pct="$e"
                current_job_completed="$f"
                speed="$g"
                avg="$h"
                eta="$i"
                last_chunk_age="$j"
                render_manifest="$k"
                render_count="$l"
                render_status="$m"
                final_output="$n"

                printf "  model: %s\n" "$model_name"
                printf "    path: %s\n" "$model_path"
                if [ -n "$total" ]; then
                    printf "    chunks: %s / %s (%s%%)\n" "$completed" "$total" "$pct"
                else
                    printf "    chunks: %s / ? (%s)\n" "$completed" "$pct"
                fi
                printf "    speed: %s | avg: %s | ETA: %s\n" "$speed" "$avg" "$eta"
                if [ -n "$current_job_completed" ] && [ "$current_job_completed" -gt 0 ] && [ "$current_job_completed" != "$completed" ]; then
                    printf "    rate basis: %s chunk(s) completed by this job\n" "$current_job_completed"
                fi
                [ -n "$last_chunk_age" ] && printf "    latest chunk age: %s\n" "$last_chunk_age"
                [ -n "$render_manifest" ] && printf "    render manifest: %s\n" "$render_manifest"
                [ -n "$render_count" ] && printf "    render count: %s\n" "$render_count"
                [ -n "$render_status" ] && printf "    render status: %s\n" "$render_status"
                [ -n "$final_output" ] && printf "    final output: %s\n" "$final_output"
                record_cluster_totals "$cluster" "$current_job_completed" "$elapsed" "$eta"
                ;;
        esac
    done <<EOF
$report
EOF

    if [ "$saw_model" -eq 0 ]; then
        printf "  (no model outputs found yet)\n"
    fi

    printf "\n"
}

trap 'exit 0' INT TERM

while :; do
    clear_screen
    printf "TTS Progress Monitor - Refreshing every %s seconds\n" "$INTERVAL"
    printf "Press Ctrl+C to exit\n"
    printf "========================================================\n"
    printf "Monitoring clusters: %s\n\n" "${REMOTE_CLUSTER_LIST[*]}"

    query_dir="$(mktemp -d "${TMPDIR:-/tmp}/tts-monitor.XXXXXX")"
    query_cluster_jobs_parallel "$query_dir"
    select_display_clusters "$query_dir"

    printf "Display mode: %s\n" "$DISPLAY_MODE"
    printf "Showing clusters: %s\n\n" "${DISPLAY_CLUSTER_LIST[*]}"

    for cluster in "${DISPLAY_CLUSTER_LIST[@]}"; do
        set_cluster_context "$cluster"
        reset_cluster_totals "$cluster"
        ROOT_SEEN=""

        JOBS_OUTPUT=""
        if [ -f "$query_dir/$cluster.jobs" ]; then
            JOBS_OUTPUT="$(cat "$query_dir/$cluster.jobs")"
        fi

        print_job_table "$JOBS_OUTPUT"

        if [ -n "$JOBS_OUTPUT" ]; then
            while IFS='|' read -r jobid job_name state elapsed node; do
                [ -n "$jobid" ] || continue

                job_detail="$(remote_job_detail "$jobid")"
                workdir="$(parse_slurm_field "$job_detail" "WorkDir")"

                roots_output=""
                if [ -n "$workdir" ]; then
                    roots_output="$(discover_run_roots "$workdir/generated")"
                fi

                if [ -z "$roots_output" ]; then
                    roots_output="$(discover_run_roots "$REMOTE_BASE/generated")"
                fi

                if [ -z "$roots_output" ]; then
                    search_base="${workdir:-$REMOTE_BASE}"
                    printf "Run tree for job %s (%s) not found yet under %s/generated\n\n" "$jobid" "$job_name" "$search_base"
                    continue
                fi

                while IFS='|' read -r root source_kind source_path source_mtime; do
                    [ -n "$root" ] || continue
                    if root_seen_contains "$root"; then
                        continue
                    fi
                    root_seen_add "$root"
                    report="$(inspect_run_root "$root" "$(elapsed_to_seconds "$elapsed")")"
                    print_root_report "$root" "$jobid" "$job_name" "$state" "$elapsed" "$node" "$source_kind" "$source_path" "$cluster" "$report"
                    if is_running_state "$state"; then
                        print_gpu_section "$jobid" "$job_name" "$state" "$job_detail"
                        printf "\n"
                    fi
                done <<EOF
$roots_output
EOF
            done <<EOF
$JOBS_OUTPUT
EOF
        else
            printf "No active %s jobs found.\n\n" "$REMOTE_CLUSTER_LABEL"

            roots_output="$(discover_run_roots "$REMOTE_BASE/generated")"
            if [ -n "$roots_output" ]; then
                while IFS='|' read -r root source_kind source_path source_mtime; do
                    [ -n "$root" ] || continue
                    if root_seen_contains "$root"; then
                        continue
                    fi
                    root_seen_add "$root"
                    report="$(inspect_run_root "$root" 0)"
                    print_root_report "$root" "n/a" "n/a" "n/a" "n/a" "n/a" "$source_kind" "$source_path" "$cluster" "$report"
                done <<EOF
$roots_output
EOF
            else
                printf "No recent run trees discovered under %s/generated.\n\n" "$REMOTE_BASE"
            fi
        fi

        print_cluster_summary "$cluster"
    done

    if [ "${#DISPLAY_CLUSTER_LIST[@]}" -gt 1 ]; then
        print_cluster_comparison
    fi

    rm -rf "$query_dir"

    printf "Next update in %s seconds... (Ctrl+C to exit)\n" "$INTERVAL"
    sleep "$INTERVAL"
done
