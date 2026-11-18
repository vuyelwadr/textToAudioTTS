#!/usr/bin/env python3
"""
Join Rich Miller audio chunks in batches to avoid memory issues
"""

import os
import glob
import torch
from pathlib import Path
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "library" / "credoMutwa" / "indaba" / "indaba1.2_rich_miller_optimized_output"

def join_audio_in_batches(input_dir, output_file="rich_miller_complete_audiobook1.2.wav", batch_size=50):
    """Join audio chunks in batches to avoid memory issues"""

    print(f"🔗 Joining Rich Miller audio chunks in batches...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size}")

    # Get all chunk files sorted numerically
    chunk_pattern = os.path.join(input_dir, "rich_miller_chunk_*.wav")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        print("❌ No chunk files found!")
        return False

    print(f"✅ Found {len(chunk_files)} chunks to join")

    try:
        print(f"🚀 Processing in batches of {batch_size} chunks...")

        # Get sample rate from first chunk
        first_chunk, sample_rate = torchaudio.load(chunk_files[0])
        if first_chunk.shape[0] > 1:
            first_chunk = torch.mean(first_chunk, dim=0, keepdim=True)

        # Process in batches
        final_audio_segments = [first_chunk]  # Start with first chunk
        total_duration = first_chunk.shape[1] / sample_rate

        print(f"📖 First chunk loaded: {os.path.basename(chunk_files[0])}")
        print(f"   Duration: {total_duration/60:.1f} minutes")
        print(f"   Sample rate: {sample_rate} Hz")

        # Process remaining chunks in batches
        remaining_files = chunk_files[1:]

        for batch_start in range(0, len(remaining_files), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_files))
            batch_files_current = remaining_files[batch_start:batch_end]

            print(f"\n📦 Processing batch {batch_start//batch_size + 1}: chunks {batch_start+2}-{batch_end+1}")

            # Load all audio chunks in this batch
            batch_audio_chunks = []
            batch_duration = 0

            for i, chunk_file in enumerate(batch_files_current):
                chunk_num = batch_start + i + 2
                print(f"📖 Loading chunk {chunk_num}/{len(chunk_files)}: {os.path.basename(chunk_file)}")

                # Load audio
                audio, _ = torchaudio.load(chunk_file)

                # Ensure mono if needed
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)

                batch_audio_chunks.append(audio)
                chunk_duration = audio.shape[1] / sample_rate
                batch_duration += chunk_duration

            print(f"   Batch duration: {batch_duration/60:.1f} minutes")

            # Concatenate all chunks in this batch
            batch_audio = torch.cat(batch_audio_chunks, dim=1)
            final_audio_segments.append(batch_audio)

            total_duration += batch_duration
            print(f"📊 Cumulative duration: {total_duration/60:.1f} minutes")

        print(f"\n🎵 Concatenating all batches...")
        # Concatenate all batch segments
        final_audio = torch.cat(final_audio_segments, dim=1)

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
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Directory with the optimized chunks
    input_dir = str(DEFAULT_INPUT_DIR)

    # Output file in the same directory
    output_file = os.path.join(input_dir, "rich_miller_complete_audiobook1.2.wav")

    # Smaller batch size to avoid memory issues
    success = join_audio_in_batches(input_dir, output_file, batch_size=25)

    if success:
        print(f"\n🎉 Rich Miller audiobook ready!")
        print(f"📂 Location: {output_file}")
        print(f"🎯 Ready to listen!")
    else:
        print(f"\n❌ Failed to create audiobook")