#!/usr/bin/env python3
"""
REAL TTS CONCURRENCY TESTER - Tests actual downloads at different concurrency levels
This runs REAL downloads against text-to-speech.online to find optimal settings
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

# Import the actual downloader
sys.path.insert(0, str(Path(__file__).parent))
from tts_parallel_fixed import download_chunk_with_detailed_errors, simple_chunk_text

# Test configurations
TEST_CONFIGS = [
    {"chunks": 5, "concurrent": 1, "chunk_size": 3000},
    {"chunks": 5, "concurrent": 3, "chunk_size": 3000},
    {"chunks": 5, "concurrent": 5, "chunk_size": 3000},
    {"chunks": 10, "concurrent": 5, "chunk_size": 3000},
    {"chunks": 10, "concurrent": 10, "chunk_size": 3000},
]

# Test text - varied lengths
TEST_TEXT = """
The sun rises over the distant mountains, casting long shadows across the valley below.
Birds begin their morning songs, filling the air with melodious chirping.
A gentle breeze rustles through the leaves of ancient oak trees.
The river flows steadily, its waters crystal clear and cool.
In the meadow, wildflowers bloom in vibrant colors - reds, yellows, and purples.
Butterflies dance from flower to flower, collecting nectar.
The sky gradually shifts from deep orange to brilliant blue as dawn breaks.
This peaceful scene repeats itself every morning, a testament to nature's eternal rhythm.
"""


async def test_real_concurrent_downloads(
    num_chunks: int,
    max_concurrent: int,
    chunk_size: int,
    output_dir: Path
) -> dict:
    """
    Test REAL downloads at specified concurrency level
    Returns performance statistics
    """
    print(f"\n{'='*80}")
    print(f"🧪 REAL TEST: {num_chunks} chunks, {max_concurrent} concurrent, {chunk_size} chars/chunk")
    print(f"{'='*80}\n")

    # Generate test text chunks
    full_text = (TEST_TEXT * 50)[:chunk_size * num_chunks]  # Generate enough text
    chunks = simple_chunk_text(full_text, chunk_size)[:num_chunks]

    semaphore = asyncio.Semaphore(max_concurrent)
    total_chunks = len(chunks)

    start_time = time.time()

    # Create download tasks
    tasks = []
    for chunk_index, chunk_text in chunks:
        async def download_with_semaphore(idx=chunk_index, text=chunk_text):
            async with semaphore:
                return await download_chunk_with_detailed_errors(
                    idx, text, output_dir, total_chunks,
                    headless=True, take_screenshots=False
                )
        tasks.append(download_with_semaphore())

    # Execute all downloads
    results = await asyncio.gather(*tasks)

    wall_clock_time = time.time() - start_time

    # Analyze results
    successful = sum(1 for _, success, _, _ in results if success)
    failed = total_chunks - successful

    successful_times = []
    for chunk_idx, success, filepath, error in results:
        if success and filepath:
            # Get file info
            try:
                file_size = Path(filepath).stat().st_size
                successful_times.append({
                    'chunk': chunk_idx,
                    'size': file_size
                })
            except:
                pass

    stats = {
        'chunks': total_chunks,
        'concurrent': max_concurrent,
        'chunk_size': chunk_size,
        'successful': successful,
        'failed': failed,
        'wall_clock_time': wall_clock_time,
        'success_rate': (successful / total_chunks * 100) if total_chunks > 0 else 0,
        'chunks_per_minute': (successful / wall_clock_time * 60) if wall_clock_time > 0 else 0,
        'avg_time_per_chunk': wall_clock_time / successful if successful > 0 else 0,
        'total_size_kb': sum(s['size'] for s in successful_times) / 1024 if successful_times else 0,
    }

    # Print results
    print(f"\n{'='*80}")
    print(f"📊 RESULTS: {num_chunks} chunks, {max_concurrent} concurrent")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{total_chunks} ({stats['success_rate']:.1f}%)")
    print(f"❌ Failed: {failed}/{total_chunks}")
    print(f"⏱️  Wall clock time: {wall_clock_time:.1f}s ({wall_clock_time/60:.2f} min)")
    print(f"📈 Average time/chunk: {stats['avg_time_per_chunk']:.1f}s")
    print(f"🚀 Throughput: {stats['chunks_per_minute']:.2f} chunks/minute")
    print(f"💾 Total downloaded: {stats['total_size_kb']:.1f} KB")
    print(f"{'='*80}\n")

    if failed > 0:
        print(f"⚠️  FAILURES:")
        for chunk_idx, success, filepath, error in results:
            if not success:
                print(f"   Chunk {chunk_idx}: {error[:200]}...")

    return stats


async def run_real_concurrency_tests():
    """
    Run comprehensive real-world tests at different concurrency levels
    """
    output_dir = Path("./real_test_output")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("🔬 REAL TTS CONCURRENT DOWNLOAD PERFORMANCE TEST")
    print("="*80)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output: {output_dir.absolute()}")
    print(f"🌐 Website: https://www.text-to-speech.online/")
    print(f"📋 Test configurations: {len(TEST_CONFIGS)}")
    print("="*80)

    all_results = []

    for config in TEST_CONFIGS:
        # Clean output dir before each test
        for f in output_dir.glob("segment_*.mp3"):
            try:
                f.unlink()
            except:
                pass

        print(f"\n⏳ Waiting 10s before next test (avoid rate limiting)...\n")
        await asyncio.sleep(10)

        try:
            stats = await test_real_concurrent_downloads(
                config['chunks'],
                config['concurrent'],
                config['chunk_size'],
                output_dir
            )
            all_results.append(stats)

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    if not all_results:
        print("\n❌ No successful test runs!")
        return

    print("\n" + "="*80)
    print("📊 FINAL SUMMARY - REAL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Chunks':<8} {'Concurrent':<12} {'Success%':<10} {'Time(s)':<10} {'Throughput':<15} {'Avg/Chunk(s)':<15}")
    print("-"*80)

    for stats in all_results:
        print(f"{stats['chunks']:<8} {stats['concurrent']:<12} "
              f"{stats['success_rate']:<10.1f} {stats['wall_clock_time']:<10.1f} "
              f"{stats['chunks_per_minute']:<15.2f} {stats['avg_time_per_chunk']:<15.1f}")

    print("="*80)

    # Find best configuration
    successful_tests = [s for s in all_results if s['success_rate'] > 80]

    if successful_tests:
        best_throughput = max(successful_tests, key=lambda x: x['chunks_per_minute'])
        most_reliable = max(all_results, key=lambda x: x['success_rate'])

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"\n   🚀 Best Throughput (>80% success):")
        print(f"      Concurrent: {best_throughput['concurrent']}")
        print(f"      Throughput: {best_throughput['chunks_per_minute']:.2f} chunks/minute")
        print(f"      Success rate: {best_throughput['success_rate']:.1f}%")

        print(f"\n   ✅ Most Reliable:")
        print(f"      Concurrent: {most_reliable['concurrent']}")
        print(f"      Success rate: {most_reliable['success_rate']:.1f}%")
        print(f"      Throughput: {most_reliable['chunks_per_minute']:.2f} chunks/minute")

        print(f"\n   📋 Recommended for production:")
        print(f"      python3 tts_parallel_fixed.py your_file.txt \\")
        print(f"          --concurrent {best_throughput['concurrent']} \\")
        print(f"          --chunk-size {best_throughput['chunk_size']}")
    else:
        print(f"\n⚠️  WARNING: No tests achieved >80% success rate!")
        print(f"   You may need to:")
        print(f"   - Reduce concurrency")
        print(f"   - Reduce chunk size")
        print(f"   - Check network connection")

    print("="*80)

    # Save results to JSON
    results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Results saved to: {results_file}")


async def main():
    print("\n" + "="*80)
    print("⚠️  IMPORTANT: This will run REAL downloads from text-to-speech.online")
    print("   This will take approximately 15-30 minutes to complete")
    print("   Make sure you have a stable internet connection")
    print("="*80)

    try:
        await run_real_concurrency_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
