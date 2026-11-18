#!/usr/bin/env python3
"""
Chunk Anna Karenina sample for multi-model TTS audit pulse
"""

import sys
import json
from pathlib import Path
from textChunkingAdapter import WebTTSChunker

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "inputs" / "anna_karenina_5min_sample.txt"
OUTPUT_DIR = REPO_ROOT / "generated" / "anna_karenina_5min_pulse"

def main():
    # Initialize chunker
    chunker = WebTTSChunker(
        max_chunk_size=1200,  # Smaller chunks for faster rendering
        min_chunk_size=300,
        output_dir=str(OUTPUT_DIR),
    )

    # Process the file
    manifest_path, plan_path = chunker.process_file(
        input_file=str(INPUT_FILE),
        prefix="anna_karenina_5min",
        num_workers=1,  # Sequential for small sample
    )

    print(f"\nChunking complete!")
    print(f"Manifest: {manifest_path}")
    print(f"Processing plan: {plan_path}")
    print(f"Chunks directory: {OUTPUT_DIR}")

    # Print manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    print(f"\nTotal chunks: {len(manifest['chunks'])}")
    for chunk in manifest['chunks']:
        print(f"  Chunk {chunk['chunk_id']}: {chunk['character_count']} chars, ~{chunk['estimated_duration_seconds']:.1f}s")

if __name__ == "__main__":
    main()