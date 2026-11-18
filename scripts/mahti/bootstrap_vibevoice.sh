#!/usr/bin/env bash
# Bootstrap VibeVoice (microsoft/VibeVoice-{1.5B,Realtime-0.5B}) for Mahti/Puhti.
#
# ⚠ CAVEATS
# 1.  Code source: Microsoft deleted their official GitHub inference repo.
#     This script clones the community mirror instead:
#     https://github.com/vibevoice-community/VibeVoice  (MIT, ≈1 k stars, 394 forks)
#     If Microsoft restores their repo, update VIBEVOICE_REPO below.
#     Original-code archive fallback: https://github.com/shijincai/VibeVoice
#
# 2.  Transformers pin: VibeVoice requires transformers==4.51.3 exactly.
#     This is INCOMPATIBLE with Higgs Audio (<4.47) – both need isolated venvs.
#     A dedicated venv named 'vibevoice' is created here.
#
# 3.  Weights license: microsoft/VibeVoice-{1.5B,Realtime-0.5B} model cards
#     say "research purpose use only".  The community fork code is MIT.
#     Do not use in commercial or production contexts.
#
# 4.  The `av` and `aiortc` packages (WebRTC/streaming, needed only for the
#     interactive Gradio demo) are intentionally skipped here.  Batch inference
#     via demo/inference_from_file.py and demo/streaming_inference_from_file.py
#     works without them.
#
# 5.  `llvmlite` / `numba` are transitive deps of librosa.  If they fail to
#     build, load a GCC module first: `module load gcc`.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# Community mirror.  Fallback: https://github.com/shijincai/VibeVoice
VIBEVOICE_REPO="https://github.com/vibevoice-community/VibeVoice.git"

_bootstrap() {
	load_pytorch_module 2.9

	VIBEVOICE_DIR="$MAHTI_EXTERNAL_ROOT/vibevoice-community"
	ensure_repo "$VIBEVOICE_REPO" "$VIBEVOICE_DIR"

	ensure_python_venv "vibevoice"
	activate_python_venv "vibevoice"

	# torch/torchaudio come from the pytorch/2.9 module (via --system-site-packages).
	# Do NOT re-install them here; only install what the module does not provide.
	#
	# transformers==4.51.3 is the exact version required by VibeVoice's internal
	# classes (VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor, etc.).
	# Later versions change the model loading API and break inference silently.
	# accelerate==1.6.0 is the version tested against transformers==4.51.3.
	python -m pip install \
		"transformers==4.51.3" \
		"accelerate==1.6.0" \
		diffusers \
		"llvmlite>=0.40.0" \
		"numba>=0.57.0" \
		scipy \
		librosa \
		ml-collections \
		absl-py \
		soundfile \
		tqdm

	# Install the vibevoice Python package (custom VibeVoice* model classes)
	# without pulling in its declared deps (managed above to preserve the
	# transformers pin and avoid torch reinstall).
	python -m pip install --no-deps -e "$VIBEVOICE_DIR"

	# Download model weights into the shared model root.
	python_hf_download \
		"microsoft/VibeVoice-1.5B" \
		"$MAHTI_MODEL_ROOT/vibevoice/microsoft--VibeVoice-1.5B"
	python_hf_download \
		"microsoft/VibeVoice-Realtime-0.5B" \
		"$MAHTI_MODEL_ROOT/vibevoice/microsoft--VibeVoice-Realtime-0.5B"

	log "VibeVoice bootstrap completed"
}

refresh_vibevoice_bootstrap_marker_if_needed() {
	local env_path="$MAHTI_ENV_ROOT/vibevoice"
	local marker="$env_path/.bootstrap-complete"
	local current_python_version existing_python_version

	current_python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
	if [[ -x "$env_path/bin/python" ]]; then
		existing_python_version="$("$env_path/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo invalid)"
		if [[ "$existing_python_version" != "$current_python_version" ]]; then
			warn "clearing stale VibeVoice bootstrap marker (python $existing_python_version -> $current_python_version)"
			rm -f "$marker"
		fi
	fi
}

load_pytorch_module 2.9
refresh_vibevoice_bootstrap_marker_if_needed
run_bootstrap_once "vibevoice" _bootstrap
activate_python_venv "vibevoice"
log "VibeVoice environment ready"
