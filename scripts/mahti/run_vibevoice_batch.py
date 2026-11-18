#!/usr/bin/env python3
"""
VibeVoice batch processor - loads model ONCE, processes ALL chunks
"""

import os
import sys
from pathlib import Path
from typing import List
import torch
import torchaudio

sys.path.insert(
    0,
    "/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/external/vibevoice-community",
)
sys.path.insert(
    0,
    "/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/external/vibevoice-community/demo",
)

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)

MODEL_PATH = "/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/models/vibevoice/microsoft--VibeVoice-Realtime-0.5B"


def generate_speech(processor, model, text: str) -> torch.Tensor:
    """Generate speech for a single text input"""
    inputs = processor(text=text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        audio_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
        )

    audio = processor.decode(audio_ids=audio_ids)
    return audio


def main():
    input_dir = Path(
        "/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-full/chunks"
    )
    output_dir = Path(
        "/scratch/project_2010549/ruwododa/textToAudioTTS-puhti/generated/audiobooks-half_of_a_yellow_sun-full/vibevoice_realtime/chunks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("chunk_*.txt"))
    total = len(txt_files)
    print(f"Found {total} text files", flush=True)

    if total == 0:
        return

    print("Loading model ONCE...", flush=True)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(MODEL_PATH)
    model = model.to("cuda")
    model.eval()
    print("Model loaded to GPU!", flush=True)

    completed = 0
    for i, txt_file in enumerate(txt_files):
        wav_file = output_dir / f"{txt_file.stem}.wav"

        if wav_file.exists() and wav_file.stat().st_size > 1000:
            completed += 1
            continue

        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                continue

            audio = generate_speech(processor, model, text)

            if isinstance(audio, torch.Tensor):
                torchaudio.save(str(wav_file), audio, sample_rate=24000)

            completed += 1
            print(f"Chunk {i + 1}/{total}: {txt_file.name}", flush=True)

        except Exception as e:
            print(f"Error {txt_file.name}: {e}", flush=True)

    print(f"Done! {completed}/{total}", flush=True)


if __name__ == "__main__":
    main()
