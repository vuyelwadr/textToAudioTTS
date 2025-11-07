#!/usr/bin/env python3
"""
MOCK TTS DOWNLOADER - Test concurrent logic without hitting real website
This verifies the concurrent download mechanism works correctly
"""

import asyncio
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import random

# Configuration
SIMULATE_DOWNLOAD_TIME_MIN = 15  # Min seconds per download
SIMULATE_DOWNLOAD_TIME_MAX = 25  # Max seconds per download
SIMULATE_FAILURE_RATE = 0.1  # 10% failure rate for realism


async def mock_download_chunk(
    chunk_index: int,
    chunk_text: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    total_chunks: int,
    stats: dict
) -> Tuple[int, bool, str, float]:
    """
    Mock download that simulates real TTS download behavior
    Returns (chunk_index, success, filepath, duration)
    """
    async with semaphore:
        start_time = time.time()

        # Simulate the download process
        download_time = random.uniform(SIMULATE_DOWNLOAD_TIME_MIN, SIMULATE_DOWNLOAD_TIME_MAX)

        print(f"[{chunk_index + 1}/{total_chunks}] Starting mock download (will take ~{download_time:.1f}s)...")

        # Simulate various stages
        await asyncio.sleep(2)  # Navigation
        print(f"[{chunk_index + 1}/{total_chunks}]   → Page loaded")

        await asyncio.sleep(1)  # Cookie consent
        print(f"[{chunk_index + 1}/{total_chunks}]   → Consent handled")

        await asyncio.sleep(1)  # Locale selection
        print(f"[{chunk_index + 1}/{total_chunks}]   → Locale selected")

        await asyncio.sleep(1)  # Voice selection
        print(f"[{chunk_index + 1}/{total_chunks}]   → Voice selected")

        await asyncio.sleep(1)  # Text entry
        print(f"[{chunk_index + 1}/{total_chunks}]   → Text entered")

        # Audio generation (longest part)
        generation_time = download_time - 8  # Remaining time after setup
        await asyncio.sleep(generation_time)
        print(f"[{chunk_index + 1}/{total_chunks}]   → Audio generated")

        await asyncio.sleep(2)  # Download
        print(f"[{chunk_index + 1}/{total_chunks}]   → Downloaded")

        # Simulate occasional failures
        if random.random() < SIMULATE_FAILURE_RATE:
            duration = time.time() - start_time
            print(f"[{chunk_index + 1}/{total_chunks}] ❌ FAILED (simulated failure)")
            stats['failed'] += 1
            return (chunk_index, False, "", duration)

        # Success - create mock file
        target_filename = f"segment_{chunk_index:05d}.mp3"
        target_path = output_dir / target_filename

        # Create fake audio file
        with open(target_path, 'wb') as f:
            # Simulate ~100KB audio file
            f.write(b'\x00' * (100 * 1024))

        duration = time.time() - start_time
        file_size = target_path.stat().st_size

        print(f"[{chunk_index + 1}/{total_chunks}] ✅ SUCCESS: {target_filename} ({file_size/1024:.1f} KB) in {duration:.1f}s")

        stats['successful'] += 1
        stats['total_time'] += duration

        return (chunk_index, True, str(target_path), duration)


async def test_concurrent_downloads(
    num_chunks: int,
    max_concurrent: int,
    output_dir: Path
) -> dict:
    """
    Test concurrent downloads and return performance stats
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Mock chunks
    chunks = [(i, f"Chunk {i} text...") for i in range(num_chunks)]

    stats = {
        'successful': 0,
        'failed': 0,
        'total_time': 0,
        'chunks': num_chunks,
        'concurrent': max_concurrent
    }

    print(f"\n{'='*70}")
    print(f"🧪 TESTING: {num_chunks} chunks, {max_concurrent} concurrent")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Create download tasks
    tasks = [
        mock_download_chunk(idx, text, output_dir, semaphore, num_chunks, stats)
        for idx, text in chunks
    ]

    # Execute all downloads
    results = await asyncio.gather(*tasks)

    wall_clock_time = time.time() - start_time

    stats['wall_clock_time'] = wall_clock_time
    stats['avg_download_time'] = stats['total_time'] / stats['successful'] if stats['successful'] > 0 else 0

    # Calculate throughput
    stats['chunks_per_minute'] = (stats['successful'] / wall_clock_time) * 60 if wall_clock_time > 0 else 0

    return stats


async def run_concurrency_tests():
    """
    Test different concurrency levels and report results
    """
    output_dir = Path("./mock_test_output")
    output_dir.mkdir(exist_ok=True)

    # Test configurations
    test_configs = [
        (10, 1),   # 10 chunks, 1 concurrent
        (10, 3),   # 10 chunks, 3 concurrent
        (10, 5),   # 10 chunks, 5 concurrent
        (20, 10),  # 20 chunks, 10 concurrent
        (20, 20),  # 20 chunks, 20 concurrent
    ]

    all_results = []

    print("="*70)
    print("🔬 CONCURRENT DOWNLOAD PERFORMANCE TEST")
    print("="*70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output: {output_dir.absolute()}")
    print(f"🎲 Simulated download time: {SIMULATE_DOWNLOAD_TIME_MIN}-{SIMULATE_DOWNLOAD_TIME_MAX}s")
    print(f"⚠️  Simulated failure rate: {SIMULATE_FAILURE_RATE*100:.0f}%")
    print("="*70)

    for num_chunks, concurrent in test_configs:
        # Clean output dir
        for f in output_dir.glob("*.mp3"):
            f.unlink()

        stats = await test_concurrent_downloads(num_chunks, concurrent, output_dir)
        all_results.append(stats)

        # Print results for this test
        print(f"\n{'='*70}")
        print(f"📊 RESULTS: {num_chunks} chunks, {concurrent} concurrent")
        print(f"{'='*70}")
        print(f"✅ Successful: {stats['successful']}/{stats['chunks']}")
        print(f"❌ Failed: {stats['failed']}/{stats['chunks']}")
        print(f"⏱️  Wall clock time: {stats['wall_clock_time']:.1f}s")
        print(f"📈 Average download time: {stats['avg_download_time']:.1f}s")
        print(f"🚀 Throughput: {stats['chunks_per_minute']:.1f} chunks/minute")

        # Calculate efficiency
        theoretical_min = (stats['avg_download_time'] * stats['chunks']) / concurrent
        efficiency = (theoretical_min / stats['wall_clock_time']) * 100 if stats['wall_clock_time'] > 0 else 0
        print(f"⚡ Efficiency: {efficiency:.1f}% (vs theoretical minimum)")
        print(f"{'='*70}\n")

        await asyncio.sleep(2)  # Pause between tests

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - CONCURRENCY COMPARISON")
    print("="*70)
    print(f"{'Chunks':<8} {'Concurrent':<12} {'Wall Time':<12} {'Throughput':<15} {'Efficiency':<12}")
    print("-"*70)

    for stats in all_results:
        theoretical_min = (stats['avg_download_time'] * stats['chunks']) / stats['concurrent']
        efficiency = (theoretical_min / stats['wall_clock_time']) * 100 if stats['wall_clock_time'] > 0 else 0

        print(f"{stats['chunks']:<8} {stats['concurrent']:<12} {stats['wall_clock_time']:<12.1f} "
              f"{stats['chunks_per_minute']:<15.1f} {efficiency:<12.1f}%")

    print("="*70)

    # Recommendation
    best_throughput = max(all_results, key=lambda x: x['chunks_per_minute'])
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Best throughput: {best_throughput['concurrent']} concurrent downloads")
    print(f"   ({best_throughput['chunks_per_minute']:.1f} chunks/minute)")
    print("="*70)


async def main():
    await run_concurrency_tests()


if __name__ == "__main__":
    asyncio.run(main())
