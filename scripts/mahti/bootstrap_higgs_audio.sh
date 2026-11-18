#!/usr/bin/env bash
# Bootstrap script for Higgs Audio v2 (bosonai/higgs-audio-v2-generation-3B-base).
# Uses the native Transformers integration from the model card instead of the
# older boson_multimodal serve engine.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

export MAHTI_ENV_ROOT="$MAHTI_WORK_ROOT/venvs"
export CSC_ENV_ROOT="$MAHTI_ENV_ROOT"

_bootstrap() {
	load_pytorch_module 2.9
	ensure_python_venv "higgs-audio"
	activate_python_venv "higgs-audio"

	# Native Transformers path from the model card. Avoid the editable
	# boson_multimodal package because it mismatches the v2 config on Puhti.
	python -m pip install \
		"transformers>=5.3.0" \
		"accelerate>=0.26.0" \
		"mistral-common>=1.11.0" \
		descript-audio-codec \
		librosa \
		soundfile \
		jinja2

	python_hf_download \
		"bosonai/higgs-audio-v2-generation-3B-base" \
		"$MAHTI_MODEL_ROOT/higgs-audio/bosonai__higgs-audio-v2-generation-3B-base"
	python_hf_download \
		"bosonai/higgs-audio-v2-tokenizer" \
		"$MAHTI_MODEL_ROOT/higgs-audio/bosonai__higgs-audio-v2-tokenizer"

	log "Higgs Audio bootstrap completed"
}

run_bootstrap_once "higgs-audio" _bootstrap
activate_python_venv "higgs-audio"
log "Higgs Audio environment ready"
