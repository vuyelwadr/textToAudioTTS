#!/usr/bin/env python3
"""
PARALLEL TTS DOWNLOADER V2 - Based on working Claude implementations
- Simple character-based chunking (700-900 chars - proven to work with TTS service)
- Working download pattern from robust_download_test.py
- Exact combining from audioChunkedKukuro.py
"""

import asyncio
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import argparse

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Error: playwright not installed")
    print("Install: pip install playwright")
    print("Then run: python3 -m playwright install chromium")
    sys.exit(1)

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("❌ Error: soundfile not installed")
    print("Install: pip install soundfile numpy")
    sys.exit(1)


# TTS Service Configuration
TTS_URL = "https://www.text-to-speech.online/"
LOCALE = "English (South Africa) [en-ZA]"
TARGET_CHUNK_SIZE = 800  # Target ~800 chars per chunk (safe for TTS service)
MAX_CHUNK_SIZE = 900     # Maximum 900 chars (TTS service limit ~1000)


def normalize_text(text: str) -> str:
    """Normalize text for chunking"""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)
    return text.strip()


def chunk_text_by_characters(text: str, target_size: int = TARGET_CHUNK_SIZE, max_size: int = MAX_CHUNK_SIZE) -> List[Tuple[int, str]]:
    """
    Simple character-based chunking with sentence boundary awareness
    Target: 700-900 chars per chunk (proven to work with web TTS service)
    """
    print(f"Chunking text: target {target_size} chars, max {max_size} chars...")

    text = normalize_text(text)

    # Split into paragraphs
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    chunks = []
    chunk_index = 0
    current_chunk = ""

    for paragraph in paragraphs:
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Would adding this sentence exceed our limit?
            test_chunk = current_chunk + (' ' if current_chunk else '') + sentence

            if len(test_chunk) > max_size and current_chunk:
                # Store current chunk and start new one
                chunks.append((chunk_index, current_chunk.strip()))
                chunk_index += 1
                current_chunk = sentence
            elif len(test_chunk) >= target_size and current_chunk:
                # At target size and at sentence boundary - good place to split
                chunks.append((chunk_index, test_chunk.strip()))
                chunk_index += 1
                current_chunk = ""
            else:
                # Keep building chunk
                current_chunk = test_chunk

    # Don't forget last chunk
    if current_chunk.strip():
        chunks.append((chunk_index, current_chunk.strip()))

    # Stats
    sizes = [len(chunk[1]) for chunk in chunks]
    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Average size: {sum(sizes) // len(sizes)} chars")
    print(f"   Min: {min(sizes)} chars, Max: {max(sizes)} chars")

    return chunks


async def download_chunk(
    chunk_index: int,
    chunk_text: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    total_chunks: int,
    headless: bool = True
):
    """Download a single chunk using working pattern from robust_download_test.py"""

    async with semaphore:
        chunk_num = f"{chunk_index:04d}"
        print(f"\n[{chunk_index + 1}/{total_chunks}] Starting chunk {chunk_num} ({len(chunk_text)} chars)")

        playwright = await async_playwright().start()

        try:
            browser = await playwright.chromium.launch(
                headless=headless,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
            )

            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            page.set_default_timeout(30000)

            # Navigate
            await page.goto(TTS_URL, wait_until="domcontentloaded")
            await asyncio.sleep(3)

            # Handle consent - try multiple selectors
            try:
                consent_selectors = [
                    'button:has-text("I agree")',
                    'button:has-text("Consent")',
                    'button:has-text("Accept")',
                    '.fc-button.fc-cta-consent'
                ]
                for selector in consent_selectors:
                    try:
                        await page.click(selector, timeout=2000)
                        await asyncio.sleep(2)
                        break
                    except:
                        continue
            except:
                pass

            # Select locale
            try:
                await page.select_option('select', LOCALE, timeout=5000)
                await asyncio.sleep(3)
            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ⚠️ Locale selection: {e}")

            # Select male voice (second select)
            try:
                selects = await page.query_selector_all('select')
                if len(selects) >= 2:
                    voice_select = selects[1]
                    options = await voice_select.query_selector_all('option')
                    for option in options:
                        text = await option.text_content()
                        if 'male' in text.lower() and 'female' not in text.lower():
                            value = await option.get_attribute('value')
                            await voice_select.select_option(value)
                            await asyncio.sleep(2)
                            break
            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ⚠️ Voice selection: {e}")

            # Enter text
            await page.fill('textarea', chunk_text)
            await asyncio.sleep(1)

            # Click Play button
            await page.click('#quick-play')
            await asyncio.sleep(10)  # Wait for generation

            # Wait for Download button to be ENABLED (not disabled)
            try:
                print(f"[{chunk_index + 1}/{total_chunks}] Waiting for download button...")
                await page.wait_for_function(
                    "document.querySelector('#download') && !document.querySelector('#download').disabled",
                    timeout=90000
                )
                await asyncio.sleep(2)

                # Get download button
                download_btn = page.locator('#download')

                # CRITICAL: Setup download promise BEFORE clicking
                async with page.expect_download(timeout=60000) as download_info:
                    # Click Download - use force to bypass overlays
                    await download_btn.click(force=True)

                # Wait for download event
                download = await download_info.value

                # Save with sequential filename
                target_path = output_dir / f"segment_{chunk_num}.mp3"
                await download.save_as(str(target_path))
                await asyncio.sleep(2)

                # Verify
                if target_path.exists():
                    file_size = target_path.stat().st_size
                    if file_size > 1000:
                        print(f"[{chunk_index + 1}/{total_chunks}] ✅ SUCCESS: {target_path.name} ({file_size:,} bytes)")
                        return (chunk_index, True, target_path)
                    else:
                        print(f"[{chunk_index + 1}/{total_chunks}] ⚠️ Small file: {file_size} bytes")
                        return (chunk_index, False, None)
                else:
                    print(f"[{chunk_index + 1}/{total_chunks}] ❌ File not found")
                    return (chunk_index, False, None)

            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ❌ Download failed: {e}")
                return (chunk_index, False, None)

        except Exception as e:
            print(f"[{chunk_index + 1}/{total_chunks}] ❌ Error: {e}")
            return (chunk_index, False, None)

        finally:
            try:
                await context.close()
                await browser.close()
                await playwright.stop()
            except:
                pass


async def download_all_chunks(chunks: List[Tuple[int, str]], output_dir: Path, max_concurrent: int = 5, headless: bool = True):
    """Download all chunks with concurrency control"""

    print(f"\n{'='*70}")
    print(f"🚀 Starting parallel downloads: {len(chunks)} chunks, {max_concurrent} concurrent")
    print(f"{'='*70}\n")

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        download_chunk(idx, text, output_dir, semaphore, len(chunks), headless)
        for idx, text in chunks
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes
    successes = sum(1 for r in results if isinstance(r, tuple) and r[1])
    failures = len(results) - successes

    print(f"\n{'='*70}")
    print(f"📊 DOWNLOAD RESULTS:")
    print(f"   ✅ Successful: {successes}/{len(chunks)}")
    print(f"   ❌ Failed: {failures}/{len(chunks)}")
    print(f"{'='*70}\n")

    return results


def combine_audio_segments(directory: Path, output_filename: str = "combined_audio.wav"):
    """
    Combine audio segments using the EXACT method from audioChunkedKukuro.py
    """
    print(f"\n{'='*70}")
    print(f"🔊 Combining audio segments from: {directory}")
    print(f"{'='*70}\n")

    # Find all segment files and sort by number
    segment_pattern = re.compile(r'segment_(\d+)\.mp3')
    segments = []

    for file in sorted(directory.glob('segment_*.mp3')):
        match = segment_pattern.search(file.name)
        if match:
            segment_num = int(match.group(1))
            segments.append((segment_num, file))

    if not segments:
        print("❌ No segment files found!")
        return None

    # Sort by segment number
    segments.sort(key=lambda x: x[0])

    print(f"Found {len(segments)} segments to combine")
    for num, file in segments[:5]:
        print(f"  segment_{num:04d}.mp3 ({file.stat().st_size:,} bytes)")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more")

    # Combine audio data
    combined_data = []
    sample_rate = None

    for num, file in segments:
        try:
            data, sr = sf.read(str(file))
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                print(f"⚠️ Warning: segment_{num:04d} has different sample rate: {sr} vs {sample_rate}")

            combined_data.append(data)
            print(f"  ✅ Added segment_{num:04d}")
        except Exception as e:
            print(f"  ❌ Error reading segment_{num:04d}: {e}")

    if not combined_data:
        print("❌ No audio data to combine!")
        return None

    # Concatenate all audio
    print(f"\n🔗 Concatenating {len(combined_data)} segments...")
    final_audio = np.concatenate(combined_data)

    # Save combined file
    output_path = directory / output_filename
    sf.write(str(output_path), final_audio, sample_rate)

    file_size = output_path.stat().st_size
    duration = len(final_audio) / sample_rate

    print(f"\n{'='*70}")
    print(f"🎉 COMBINED AUDIO SAVED!")
    print(f"   📍 File: {output_path}")
    print(f"   📊 Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    print(f"   ⏱️ Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
    print(f"   🎵 Sample rate: {sample_rate} Hz")
    print(f"{'='*70}\n")

    return output_path


async def main():
    parser = argparse.ArgumentParser(description='Parallel TTS Downloader V2')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--output-dir', default='tts_output', help='Output directory')
    parser.add_argument('--concurrent', type=int, default=5, help='Number of concurrent downloads')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--no-combine', action='store_true', help='Skip combining audio segments')

    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"🎯 PARALLEL TTS DOWNLOADER V2")
    print(f"{'='*70}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Input: {args.input_file}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔄 Concurrent: {args.concurrent}")
    print(f"👁️ Headless: {args.headless}")
    print(f"{'='*70}\n")

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    text = input_path.read_text(encoding='utf-8')
    print(f"📝 Read {len(text)} characters ({len(text.split())} words)\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir.absolute()}\n")

    # Chunk text
    chunks = chunk_text_by_characters(text, TARGET_CHUNK_SIZE, MAX_CHUNK_SIZE)

    # Download all chunks
    results = await download_all_chunks(chunks, output_dir, args.concurrent, args.headless)

    # Combine audio segments
    if not args.no_combine:
        combined_path = combine_audio_segments(output_dir)
        if combined_path:
            print(f"✅ SUCCESS: Combined audio saved to {combined_path}")
        else:
            print(f"❌ FAILED: Could not combine audio segments")

    print(f"\n{'='*70}")
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
