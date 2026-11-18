#!/usr/bin/env python3
"""
Chunk the Indaba text for parallel TTS processing
"""

import sys
import json
from pathlib import Path
from textChunkingAdapter import WebTTSChunker


def main():
    # Configuration
    input_file = Path(
        "/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/library/credoMutwa/indaba/indaba1.1.txt"
    )
    output_dir = Path(
        "/Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/library/credoMutwa/indaba/ttsOnline"
    )

    # Initialize chunker with web-optimized settings
    chunker = WebTTSChunker(
        max_chunk_size=5000,  # Reduced for better quality
        min_chunk_size=1000,
        output_dir=str(output_dir),
    )

    # Process the file
    manifest_path, plan_path = chunker.process_file(
        input_file=str(input_file),
        prefix="indaba1.1",
        num_workers=8,  # Use 8 parallel workers
    )

    print(f"\nChunking complete!")
    print(f"Manifest: {manifest_path}")
    print(f"Processing plan: {plan_path}")
    print(f"Chunks directory: {output_dir}")


if __name__ == "__main__":
    main()
