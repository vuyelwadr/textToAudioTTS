#!/bin/bash
#SBATCH --job-name=tts-heavy
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch --account <project> scripts/mahti/run_model.slurm <model-id> [runner args...]" >&2
    exit 2
fi

MODEL_ID="$1"
shift

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/mahti/common.sh" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/scripts/mahti"
else
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

if [[ -n "${LOCAL_SCRATCH:-}" ]]; then
    log "LOCAL_SCRATCH available at $LOCAL_SCRATCH"
    if [[ "${CSC_PREFER_LOCAL_SCRATCH:-0}" == "1" ]]; then
        export TMPDIR="$LOCAL_SCRATCH"
        export TMP="$TMPDIR"
        export TEMP="$TMPDIR"
        log "Using LOCAL_SCRATCH for TMPDIR because CSC_PREFER_LOCAL_SCRATCH=1"
    fi
fi

run_python() {
    srun python "$@"
}

case "$MODEL_ID" in
    orpheus)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_orpheus.sh"
        activate_python_venv "orpheus-tts"
        run_python "$SCRIPT_DIR/run_orpheus.py" "$@"
        ;;
    fish-speech)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_fish_speech.sh"
        activate_python_venv "fish-speech"
        run_python "$SCRIPT_DIR/run_fish_speech.py" "$@"
        ;;
    fish-s2pro)
        load_pytorch_module 2.9
        # bootstrap_fish_s2pro.sh applies this same override internally
        # (projappl is at 100% inode utilization), but that only affects its
        # own subshell -- the parent shell here (which calls
        # activate_python_venv and later run_python) needs it too.
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_fish_s2pro.sh"
        activate_python_venv "fish-s2pro"
        run_python "$SCRIPT_DIR/run_fish_s2pro.py" "$@"
        ;;
    f5-tts)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_f5_tts.sh"
        activate_python_venv "f5-tts"
        run_python "$SCRIPT_DIR/run_f5_tts.py" "$@"
        ;;
    xtts)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export COQUI_TOS_AGREED="${COQUI_TOS_AGREED:-1}"
        bash "$SCRIPT_DIR/bootstrap_xtts.sh"
        activate_python_venv "xtts-v2"
        run_python "$SCRIPT_DIR/run_xtts.py" "$@"
        ;;
    gpt-sovits)
        load_pytorch_module 2.7
        bash "$SCRIPT_DIR/bootstrap_gpt_sovits.sh"
        activate_python_venv "gpt-sovits"
        run_python "$SCRIPT_DIR/run_gpt_sovits.py" "$@"
        ;;
    openvoice-v2)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_openvoice_v2.sh"
        activate_python_venv "openvoice-v2"
        run_python "$SCRIPT_DIR/run_openvoice_v2.py" "$@"
        ;;
    omnivoice)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_omnivoice.sh"
        activate_python_venv "omnivoice"
        run_python "$SCRIPT_DIR/run_omnivoice.py" "$@"
        ;;
    voxcpm2)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_voxcpm2.sh"
        activate_python_venv "voxcpm2"
        run_python "$SCRIPT_DIR/run_voxcpm2.py" "$@"
        ;;
    dia)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_dia.sh"
        activate_python_venv "dia"
        run_python "$SCRIPT_DIR/run_dia.py" "$@"
        ;;
    parler-tts-large)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_parler_tts_large.sh"
        activate_python_venv "parler-tts-large"
        run_python "$SCRIPT_DIR/run_parler_tts_large.py" "$@"
        ;;
    spark-tts)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_spark_tts.sh"
        activate_python_venv "spark-tts"
        run_python "$SCRIPT_DIR/run_spark_tts.py" "$@"
        ;;
    csm-1b)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_csm_1b.sh"
        activate_python_venv "csm-1b"
        run_python "$SCRIPT_DIR/run_csm_1b.py" "$@"
        ;;
    qwen3-tts)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_qwen3_tts.sh"
        activate_python_venv "qwen3-tts"
        run_python "$SCRIPT_DIR/run_qwen3_tts.py" "$@"
        ;;
    faster-qwen3-tts)
        load_pytorch_module 2.9
        bash "$SCRIPT_DIR/bootstrap_faster_qwen3_tts.sh"
        activate_python_venv "faster-qwen3-tts"
        run_python "$SCRIPT_DIR/run_faster_qwen3_tts.py" "$@"
        ;;
    breeze-tts2)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_breeze_tts2.sh"
        activate_python_venv "breeze-tts2"
        run_python "$SCRIPT_DIR/run_breeze_tts2.py" "$@"
        ;;
    cosyvoice3)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_cosyvoice3.sh"
        activate_python_venv "cosyvoice3"
        run_python "$SCRIPT_DIR/run_cosyvoice3.py" "$@"
        ;;
    higgs-audio-v3)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_higgs_audio_v3.sh"
        activate_python_venv "higgs-audio-v3"
        run_python "$SCRIPT_DIR/run_higgs_audio_v3.py" "$@"
        ;;
    kyutai-tts)
        load_pytorch_module 2.9
        # /projappl/$CSC_PROJECT is at 100% inode utilization (same issue
        # noted for fish-s2pro/step-audio-editx above), but that only affects
        # its own subshell -- the parent shell here (which calls
        # activate_python_venv and later run_python) needs it too, or it
        # would look for the venv/repo under the projappl-backed default
        # instead of where bootstrap actually put them on scratch.
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_kyutai_tts.sh"
        activate_python_venv "kyutai-tts"
        run_python "$SCRIPT_DIR/run_kyutai_tts.py" "$@"
        ;;
    moss-tts)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_moss_tts.sh"
        activate_python_venv "moss-tts"
        run_python "$SCRIPT_DIR/run_moss_tts.py" "$@"
        ;;
    kugelaudio-open)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_kugelaudio_open.sh"
        activate_python_venv "kugelaudio-open"
        run_python "$SCRIPT_DIR/run_kugelaudio_open.py" "$@"
        ;;
    echo-tts)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_echo_tts.sh"
        activate_python_venv "echo-tts"
        run_python "$SCRIPT_DIR/run_echo_tts.py" "$@"
        ;;
    higgs-audio)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_higgs_audio.sh"
        activate_python_venv "higgs-audio"
        run_python "$SCRIPT_DIR/run_higgs_audio.py" "$@"
        ;;
    chatterbox-turbo)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_chatterbox_turbo.sh"
        activate_python_venv "chatterbox-turbo"
        run_python "$SCRIPT_DIR/run_chatterbox_turbo.py" "$@"
        ;;
    voxtral)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_voxtral.sh"
        activate_python_venv "voxtral"
        run_python "$SCRIPT_DIR/run_voxtral.py" "$@"
        ;;
    vibevoice)
        load_pytorch_module 2.9
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        bash "$SCRIPT_DIR/bootstrap_vibevoice.sh"
        activate_python_venv "vibevoice"
        run_python "$SCRIPT_DIR/run_vibevoice.py" "$@"
        ;;
    step-audio-editx)
        load_pytorch_module 2.9
        # bootstrap_step_audio_editx.sh applies this same override internally,
        # but that only affects its own subshell -- the parent shell here
        # (which calls activate_python_venv and later run_python) needs it
        # too, or it would look for the venv/repo under the projappl-backed
        # default instead of where bootstrap actually put them on scratch.
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_EXTERNAL_ROOT="$MAHTI_WORK_ROOT/external"
        export CSC_EXTERNAL_ROOT="$MAHTI_EXTERNAL_ROOT"
        bash "$SCRIPT_DIR/bootstrap_step_audio_editx.sh"
        activate_python_venv "step-audio-editx"
        run_python "$SCRIPT_DIR/run_step_audio_editx.py" "$@"
        ;;
    tada)
        # hume-tada pins torch>=2.7.0,<2.8.0, so load pytorch/2.7 instead of
        # the repo-wide default of 2.9 (see bootstrap_tada.sh for details).
        load_pytorch_module 2.7
        # bootstrap_tada.sh applies this same override internally, but that
        # only affects its own subshell -- the parent shell here (which calls
        # activate_python_venv and later run_python) needs it too, or it
        # would look for the venv under the projappl-backed default instead
        # of where bootstrap actually put it on scratch.
        export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
        export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"
        export MAHTI_LOCK_ROOT="$MAHTI_WORK_ROOT/.locks"
        mkdir -p "$MAHTI_LOCK_ROOT"
        bash "$SCRIPT_DIR/bootstrap_tada.sh"
        activate_python_venv "tada"
        run_python "$SCRIPT_DIR/run_tada.py" "$@"
        ;;
    inworld-tts-poc)
        bash "$SCRIPT_DIR/bootstrap_inworld_tts.sh"
        # shellcheck disable=SC1091
        source "$MAHTI_EXTERNAL_ROOT/inworld-tts/.venv/bin/activate"
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        # Keep Lightning temp checkpoint files on scratch so save_checkpoint()
        # can rename in place instead of copying multi-GB artifacts from local NVMe.
        export TMPDIR="${MAHTI_CACHE_ROOT}/tmp"
        export TMP="$TMPDIR"
        export TEMP="$TMPDIR"
        mkdir -p "$TMPDIR"
        # The Inworld wrapper launches its own distributed srun steps for vectorization.
        # Running the wrapper itself under srun deadlocks those inner steps on Mahti.
        python "$SCRIPT_DIR/run_inworld_tts_poc.py" "$@"
        ;;
    *)
        echo "unknown model id: $MODEL_ID" >&2
        echo "expected one of: orpheus, fish-speech, fish-s2pro, f5-tts, xtts, gpt-sovits, openvoice-v2, omnivoice, voxcpm2, dia, parler-tts-large, spark-tts, csm-1b, qwen3-tts, faster-qwen3-tts, kyutai-tts, moss-tts, kugelaudio-open, echo-tts, higgs-audio, chatterbox-turbo, voxtral, vibevoice, tada, step-audio-editx, breeze-tts2, cosyvoice3, higgs-audio-v3, inworld-tts-poc" >&2
        exit 2
        ;;
esac
