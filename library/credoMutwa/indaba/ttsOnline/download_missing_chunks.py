#!/usr/bin/env python3
"""
Download only missing chunks and combine all audio
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse
import re
import numpy as np
import soundfile as sf

# Import from parallel_tts_v2
sys.path.insert(0, str(Path(__file__).parent))
from parallel_tts_v2 import download_chunk, combine_audio_segments


def find_missing_chunks(output_dir: Path, expected_total: int):
    """Find which chunks are missing"""
    segment_pattern = re.compile(r'segment_(\d+)\.mp3')

    existing = []
    for file in output_dir.glob('segment_*.mp3'):
        match = segment_pattern.search(file.name)
        if match:
            existing.append(int(match.group(1)))

    expected = set(range(expected_total))
    actual = set(existing)
    missing = sorted(expected - actual)

    return missing, len(existing)


async def download_missing(missing_indices, chunks_text, output_dir, headless=True):
    """Download only the missing chunks"""

    if not missing_indices:
        print("✅ No missing chunks - all downloads complete!")
        return True

    print(f"\n{'='*70}")
    print(f"📥 Downloading {len(missing_indices)} missing chunks")
    print(f"{'='*70}\n")

    # Create tasks for missing chunks
    semaphore = asyncio.Semaphore(5)  # Lower concurrency for retries

    tasks = []
    for idx in missing_indices:
        if idx < len(chunks_text):
            chunk_text = chunks_text[idx][1]
            tasks.append(download_chunk(idx, chunk_text, output_dir, semaphore, len(chunks_text), headless))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if isinstance(r, tuple) and r[1])
    failures = len(results) - successes

    print(f"\n{'='*70}")
    print(f"📊 RETRY RESULTS:")
    print(f"   ✅ Successful: {successes}/{len(missing_indices)}")
    print(f"   ❌ Failed: {failures}/{len(missing_indices)}")
    print(f"{'='*70}\n")

    return failures == 0


async def main():
    parser = argparse.ArgumentParser(description='Download missing chunks and combine')
    parser.add_argument('input_file', help='Original input text file')
    parser.add_argument('--output-dir', default='indaba', help='Output directory')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return 1

    # Read and chunk the input text (same way as main script)
    from parallel_tts_v2 import chunk_text_by_characters, normalize_text

    input_path = Path(args.input_file)
    text = input_path.read_text(encoding='utf-8')
    chunks = chunk_text_by_characters(text)

    print(f"📊 Total chunks expected: {len(chunks)}")

    # Find missing chunks
    missing, existing = find_missing_chunks(output_dir, len(chunks))

    print(f"✅ Existing segments: {existing}")
    print(f"❌ Missing segments: {len(missing)}")

    if missing:
        print(f"\nMissing chunk indices: {missing}")

        # Download missing chunks
        success = await download_missing(missing, chunks, output_dir, args.headless)

        # Check again
        missing_after, existing_after = find_missing_chunks(output_dir, len(chunks))

        if missing_after:
            print(f"\n⚠️ Still missing {len(missing_after)} chunks after retry:")
            for m in missing_after:
                print(f"   segment_{m:04d}.mp3")
            print(f"\n⚠️ Combining {existing_after}/{len(chunks)} available segments anyway...")
        else:
            print(f"\n🎉 All chunks downloaded successfully!")

    # Combine all available segments
    combined_path = combine_audio_segments(output_dir)

    if combined_path:
        print(f"\n✅ SUCCESS: Combined audio at {combined_path}")
        return 0
    else:
        print(f"\n❌ FAILED: Could not combine audio")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
