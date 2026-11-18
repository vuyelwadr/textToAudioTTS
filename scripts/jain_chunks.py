#!/usr/bin/env python3
"""
Join all Rich Miller audio chunks into final audiobook
"""

import os
import glob
import torch
from pathlib import Path
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "library" / "credoMutwa" / "indaba" / "indaba1.2_rich_miller_optimized_output"

def join_audio_chunks(input_dir, output_file="rich_miller_complete_audiobook.wav"):
    """Join all audio chunks into one file using torchaudio"""

    print(f"🔗 Joining Rich Miller audio chunks...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")

    # Get all chunk files sorted numerically
    chunk_pattern = os.path.join(input_dir, "rich_miller_chunk_*.wav")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        print("❌ No chunk files found!")
        return False

    print(f"✅ Found {len(chunk_files)} chunks to join")

    try:
        print(f"🚀 Loading and concatenating chunks...")

        # Load all audio chunks
        audio_chunks = []
        total_duration = 0

        for i, chunk_file in enumerate(chunk_files):
            print(f"📖 Loading chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")

            # Load audio
            audio, sample_rate = torchaudio.load(chunk_file)

            # Ensure mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            audio_chunks.append(audio)
            chunk_duration = audio.shape[1] / sample_rate
            total_duration += chunk_duration

            print(f"   Duration: {chunk_duration:.1f}s")

        print(f"📊 Total audio duration: {total_duration/60:.1f} minutes")
        print(f"🎵 Concatenating all chunks...")

        # Concatenate all chunks
        final_audio = torch.cat(audio_chunks, dim=1)

        print(f"💾 Saving final audiobook...")

        # Save the joined audio
        torchaudio.save(output_file, final_audio, sample_rate)

        # Get file size
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✅ Successfully joined all chunks!")
        print(f"📁 Final audiobook: {output_file}")
        print(f"📊 File size: {output_size:.1f} MB")
        print(f"📈 Total chunks: {len(chunk_files)}")
        print(f"⏱️  Total duration: {total_duration/60:.1f} minutes")

        return True

    except Exception as e:
        print(f"❌ Error joining chunks: {e}")
        return False

if __name__ == "__main__":
    # Directory with the optimized chunks
    input_dir = str(DEFAULT_INPUT_DIR)

    # Output file in the same directory
    output_file = os.path.join(input_dir, "rich_miller_complete_audiobook1.2.wav")

    success = join_audio_chunks(input_dir, output_file)

    if success:
        print(f"\n🎉 Rich Miller audiobook ready!")
        print(f"📂 Location: {output_file}")
        print(f"🎯 Ready to listen!")
    else:
        print(f"\n❌ Failed to create audiobook")