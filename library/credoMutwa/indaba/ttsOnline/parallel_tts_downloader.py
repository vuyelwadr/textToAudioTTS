#!/usr/bin/env python3
"""
PARALLEL TTS DOWNLOADER - Download TTS audio in chunks with parallel processing
Uses EXACT chunking from audioChunkedKukuro.py (phoneme-based)
Uses EXACT combining from audioChunkedKukuro.py
"""

import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Error: playwright not installed")
    print("Install: pip install playwright")
    print("Then run: python3 -m playwright install chromium")
    sys.exit(1)

# Optional: Import Kokoro for phoneme-based chunking (not needed for web TTS)
# Web TTS has ~1000 character limit, so we use simple character chunking
try:
    from kokoro import KPipeline
    from kokoro.model import KModel
    from misaki import en, espeak
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    # Not critical - we'll use character-based chunking for web TTS

# Configuration
MAX_CONCURRENT_DOWNLOADS = 20  # Download 20 chunks in parallel
HEADLESS_MODE = True  # Run browser in headless mode
LOCALE = "English (South Africa) [en-ZA]"
SAMPLE_RATE = 24000  # Standard TTS output sample rate
LANG_CODE = 'a'  # 'a' for English (used by Kokoro)

def normalize_text(text):
    """
    Normalize text to handle line breaks more naturally (FROM audioChunkedKukuro.py)
    1. Join lines that are part of the same sentence
    2. Preserve paragraph breaks (double newlines)
    3. Remove unnecessary whitespace
    """
    # First, preserve paragraph breaks by temporarily replacing them
    text = text.replace('\n\n', '[PARAGRAPH_BREAK]')

    # Join lines that are part of the same sentence
    # Replace single newlines with spaces unless they follow punctuation
    text = re.sub(r'(?<![.!?])\n', ' ', text)

    # Restore paragraph breaks
    text = text.replace('[PARAGRAPH_BREAK]', '\n\n')

    # Fix any excess whitespace
    text = re.sub(r' +', ' ', text)

    return text.strip()

def simple_char_chunk(text: str, max_chars: int = 900) -> List[Tuple[int, str]]:
    """
    Simple character-based chunking for WEB TTS service (has ~1000 char limit)
    Chunks at sentence boundaries to preserve semantic meaning
    Returns list of (chunk_index, chunk_text) tuples
    """
    print(f"Chunking text using character limit ({max_chars} chars per chunk for web service)...")

    # Normalize text
    text = normalize_text(text)

    # Split at paragraph boundaries first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Try adding this sentence to current chunk
            temp_chunk = current_chunk + (' ' if current_chunk else '') + sentence

            # Check if adding this sentence would exceed limit
            if len(temp_chunk) > max_chars and current_chunk:
                # Store current chunk and start new one
                chunks.append((chunk_index, current_chunk.strip()))
                chunk_index += 1
                current_chunk = sentence
            else:
                # Keep adding to current chunk
                current_chunk = temp_chunk

    # Don't forget the last chunk
    if current_chunk:
        chunks.append((chunk_index, current_chunk.strip()))

    print(f"Created {len(chunks)} chunks (character-based for web TTS)")
    return chunks

def improved_chunk_text(text: str, lang_code: str = 'a', max_phonemes: int = 200) -> List[Tuple[int, str]]:
    """
    Phoneme-based chunking adapted for WEB TTS service
    KPipeline default = 510 phonemes, audioChunkedKukuro.py = 410 phonemes
    For web TTS with ~1000 char limit, we use 200 phonemes = ~600-1000 chars
    Returns list of (chunk_index, chunk_text) tuples
    """
    pipeline = KPipeline(lang_code=lang_code, model=False)
    pipeline.g2p.nlp.max_length = 5000000

    print(f"Chunking text using phoneme algorithm ({max_phonemes} phoneme limit for web TTS)...")

    # Normalize the text to handle line breaks properly
    text = normalize_text(text)

    # First split at paragraph boundaries
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    chunks = []
    chunk_index = 0

    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""
        current_phonemes = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Try adding this sentence to current chunk
            temp_chunk = current_chunk + ('' if not current_chunk else ' ') + sentence
            temp_phonemes = pipeline.g2p(temp_chunk)

            # Check if adding this sentence would exceed our limit
            if len(temp_phonemes) > max_phonemes and current_chunk:
                # Store current chunk and start a new one
                chunks.append((chunk_index, current_chunk))
                chunk_index += 1
                current_chunk = sentence
                current_phonemes = pipeline.g2p(sentence)
            else:
                # Keep adding to current chunk
                current_chunk = temp_chunk
                current_phonemes = temp_phonemes

        # Don't forget the last chunk
        if current_chunk:
            chunks.append((chunk_index, current_chunk))
            chunk_index += 1

    print(f"Created {len(chunks)} optimized text chunks (phoneme-based)")
    return chunks

async def download_chunk(
    chunk_index: int,
    chunk_text: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    total_chunks: int,
    headless_mode: bool = True
) -> Tuple[int, bool, str]:
    """
    Download a single TTS audio chunk
    Returns (chunk_index, success, filepath)
    """
    async with semaphore:  # Limit concurrent downloads
        playwright = None
        browser = None
        context = None

        try:
            print(f"[{chunk_index + 1}/{total_chunks}] Starting download...")

            # Start Playwright
            playwright = await async_playwright().start()

            # Launch browser in headless mode
            browser = await playwright.chromium.launch(
                headless=headless_mode,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
            )

            # Create context with downloads enabled
            context = await browser.new_context(
                accept_downloads=True,
                viewport={"width": 1280, "height": 800}
            )

            page = await context.new_page()
            page.set_default_timeout(30000)

            # Navigate to website
            await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
            await asyncio.sleep(2)

            # Handle consent dialog
            try:
                consent_selectors = [
                    'button:has-text("Consent")',
                    '.fc-button.fc-cta-consent',
                    'button:has-text("I agree")'
                ]

                for selector in consent_selectors:
                    try:
                        consent_btn = page.locator(selector).first
                        if await consent_btn.is_visible(timeout=2000):
                            await consent_btn.click()
                            await asyncio.sleep(2)
                            break
                    except:
                        continue
            except:
                pass

            # Select locale: English (South Africa)
            try:
                locale_select = page.locator('select').first
                await locale_select.wait_for(state='visible', timeout=10000)
                await locale_select.select_option(label=LOCALE)
                await asyncio.sleep(5)  # Wait for voice options to reload
            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ⚠️ Locale selection failed: {e}")

            # Select male voice
            try:
                selects = await page.locator('select').all()
                if len(selects) >= 2:
                    voice_select = selects[1]
                    options = await voice_select.locator('option').all()

                    for option in options:
                        text = await option.text_content()
                        text_lower = text.lower()
                        is_male = any(keyword in text_lower for keyword in ['male', 'guy', 'man', 'luke'])
                        is_female = 'female' in text_lower

                        if is_male and not is_female:
                            value = await option.get_attribute('value')
                            await voice_select.select_option(value=value)
                            await asyncio.sleep(1)
                            break
            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ⚠️ Voice selection failed: {e}")

            # Enter text
            textarea = page.locator('textarea').first
            await textarea.wait_for(state='visible', timeout=10000)
            await textarea.clear()
            await textarea.fill(chunk_text)
            print(f"[{chunk_index + 1}/{total_chunks}] Entered {len(chunk_text)} chars")
            await asyncio.sleep(1)

            # Generate speech
            play_btn = page.locator('#quick-play')
            await play_btn.wait_for(state='visible', timeout=10000)
            await play_btn.click()
            print(f"[{chunk_index + 1}/{total_chunks}] Clicked Play button")

            # Wait for speech generation (longer chunks need more time)
            await asyncio.sleep(20)
            print(f"[{chunk_index + 1}/{total_chunks}] Waited 20s for generation")

            # Wait for download button to be enabled (increased timeout for longer texts)
            print(f"[{chunk_index + 1}/{total_chunks}] Waiting for Download button to enable...")
            await page.wait_for_function(
                "document.querySelector('#download') && !document.querySelector('#download').disabled",
                timeout=90000
            )
            print(f"[{chunk_index + 1}/{total_chunks}] Download button enabled!")
            await asyncio.sleep(2)

            # Download audio
            download_btn = page.locator('#download')
            await download_btn.wait_for(state='visible', timeout=10000)

            async with page.expect_download(timeout=60000) as download_info:
                await download_btn.click()

            download = await download_info.value

            # Save with ordered filename
            target_filename = f"segment_{chunk_index:05d}.mp3"
            target_path = output_dir / target_filename

            await download.save_as(str(target_path))
            await asyncio.sleep(1)

            # Verify download
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size > 1000:
                    print(f"[{chunk_index + 1}/{total_chunks}] ✅ Downloaded: {target_filename} ({file_size/1024:.1f} KB)")
                    return (chunk_index, True, str(target_path))
                else:
                    print(f"[{chunk_index + 1}/{total_chunks}] ❌ File too small: {file_size} bytes")
                    return (chunk_index, False, "")
            else:
                print(f"[{chunk_index + 1}/{total_chunks}] ❌ File not found")
                return (chunk_index, False, "")

        except Exception as e:
            print(f"\n[{chunk_index + 1}/{total_chunks}] ❌ Download failed:")
            print(f"   Error: {e}")
            print(f"   Chunk length: {len(chunk_text)} chars")
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()
            print()
            return (chunk_index, False, "")

        finally:
            # Cleanup
            try:
                if context:
                    await context.close()
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()
            except:
                pass

async def download_all_chunks(
    chunks: List[Tuple[int, str]],
    output_dir: Path,
    max_concurrent: int = 20,
    headless_mode: bool = True
) -> List[str]:
    """
    Download all chunks in parallel with concurrency limit
    Returns list of successful filepaths
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total_chunks = len(chunks)

    print(f"\n🚀 Starting parallel downloads: {total_chunks} chunks, {max_concurrent} concurrent")
    print("=" * 70)

    # Create download tasks
    tasks = [
        download_chunk(chunk_index, chunk_text, output_dir, semaphore, total_chunks, headless_mode)
        for chunk_index, chunk_text in chunks
    ]

    # Execute all downloads
    results = await asyncio.gather(*tasks)

    # Collect successful downloads
    successful_files = []
    failed_indices = []

    for chunk_index, success, filepath in results:
        if success:
            successful_files.append(filepath)
        else:
            failed_indices.append(chunk_index)

    print("\n" + "=" * 70)
    print(f"✅ Successful: {len(successful_files)}/{total_chunks}")
    if failed_indices:
        print(f"❌ Failed: {len(failed_indices)} chunks: {failed_indices}")

    return successful_files

def combine_audio_segments(directory: Path, output_filename: str = "combined_audio.wav") -> str:
    """
    Combine all audio segments in order (similar to audioChunkedKukuro.py)
    Returns path to combined audio file
    """
    try:
        print(f"\n📦 Combining audio segments...")
        print("=" * 70)

        # Find all segment files
        segment_files = sorted([f for f in directory.glob("segment_*.mp3")])

        if not segment_files:
            print("❌ No audio segments found")
            return None

        print(f"Found {len(segment_files)} audio segments")

        # Sort by segment number to ensure correct order
        segment_files.sort(key=lambda x: int(re.search(r'segment_(\d+)', x.name).group(1)))

        print(f"First file: {segment_files[0].name}")
        print(f"Last file: {segment_files[-1].name}")

        # Load and combine audio
        combined_audio = []
        sample_rate = None

        for filepath in segment_files:
            try:
                audio, sr = sf.read(str(filepath))
                combined_audio.append(audio)
                if sample_rate is None:
                    sample_rate = sr
            except Exception as e:
                print(f"⚠️ Error reading {filepath.name}: {e}")

        if combined_audio:
            # Concatenate all audio segments
            combined_audio = np.concatenate(combined_audio)

            # Save combined audio
            combined_filepath = directory / output_filename
            sf.write(str(combined_filepath), combined_audio, sample_rate or SAMPLE_RATE)

            file_size = combined_filepath.stat().st_size
            duration = len(combined_audio) / (sample_rate or SAMPLE_RATE)

            print(f"\n✅ Combined audio saved!")
            print(f"📄 File: {combined_filepath}")
            print(f"📊 Size: {file_size/1024/1024:.1f} MB")
            print(f"⏱️ Duration: {duration/60:.1f} minutes")

            return str(combined_filepath)
        else:
            print("❌ No audio segments loaded successfully")
            return None

    except Exception as e:
        print(f"❌ Error combining audio: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel TTS downloader with phoneme-based chunking")
    parser.add_argument("input_file", type=str, help="Input text file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--concurrent", type=int, default=20,
                        help="Max concurrent downloads (default: 20)")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run in headless mode (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser windows")
    parser.add_argument("--no-combine", action="store_true", default=False,
                        help="Don't combine segments after download")

    args = parser.parse_args()

    # Use argument values
    max_concurrent = args.concurrent
    headless_mode = args.headless

    print("=" * 70)
    print("🎯 PARALLEL TTS DOWNLOADER - PHONEME-BASED CHUNKING")
    print("=" * 70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Input file: {args.input_file}")
    if KOKORO_AVAILABLE:
        print(f"📦 Chunking: 140 phonemes per chunk (adapted for web TTS ~1000 char limit)")
    else:
        print(f"📦 Chunking: 900 chars per chunk (character-based fallback)")
    print(f"🔄 Concurrent downloads: {max_concurrent}")
    print(f"👁️ Headless mode: {headless_mode}")
    print()

    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input_file}")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"📝 Read {len(text)} characters ({len(text.split())} words)")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate based on input filename
        base_name = input_path.stem
        output_dir = input_path.parent / f"{base_name}_tts_output"

    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()

    # Chunk the text using PHONEME-BASED chunking (like audioChunkedKukuro.py)
    # BUT with reduced phoneme limit to fit web TTS ~1000 char constraint
    # KPipeline default=510, audioChunkedKukuro=410, web TTS needs=140 phonemes (~700-900 chars)
    if KOKORO_AVAILABLE:
        MAX_PHONEMES = 140  # Reduced from 410 to fit web TTS ~1000 char limit
        chunks = improved_chunk_text(text, LANG_CODE, MAX_PHONEMES)
    else:
        # Fallback to character chunking if Kokoro not available
        MAX_CHARS = 900
        chunks = simple_char_chunk(text, MAX_CHARS)

    print(f"\n📊 Processing {len(chunks)} chunks")
    print(f"   Average chunk size: {len(text) / len(chunks):.0f} chars")
    print()

    # Download all chunks in parallel
    successful_files = await download_all_chunks(chunks, output_dir, max_concurrent, headless_mode)

    if not successful_files:
        print("\n❌ No chunks downloaded successfully")
        return 1

    # Combine audio segments
    if not args.no_combine:
        combined_file = combine_audio_segments(output_dir, "combined_audio.wav")
        if combined_file:
            print("\n" + "=" * 70)
            print("🎉 SUCCESS!")
            print("=" * 70)
            print(f"✅ Downloaded {len(successful_files)} chunks")
            print(f"✅ Combined into: {combined_file}")
        else:
            print("\n⚠️ Downloads completed but combining failed")
            return 1
    else:
        print("\n" + "=" * 70)
        print("✅ Downloads completed (not combined)")
        print("=" * 70)

    print()
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
