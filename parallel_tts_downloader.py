#!/usr/bin/env python3
"""
PARALLEL TTS DOWNLOADER - Working version with detailed error reporting
- Simple chunking for reliability
- Detailed step-by-step error reporting
- Concurrent download support
- Test mode for validation
"""

import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import argparse

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("❌ Error: playwright not installed")
    print("Install: pip install playwright")
    print("Then run: playwright install chromium")
    sys.exit(1)

# Configuration
TTS_URL = "https://www.text-to-speech.online/"
LOCALE = "English (South Africa)"
SAMPLE_RATE = 24000
MAX_RETRIES = 2


def simple_chunk_text(text: str, chunk_size: int = 3000) -> List[Tuple[int, str]]:
    """
    Simple character-based chunking for reliability
    Tries to break at sentence boundaries when possible
    """
    chunks = []
    chunk_index = 0

    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)

        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append((chunk_index, current_chunk.strip()))
                chunk_index += 1
                current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append((chunk_index, current_chunk.strip()))

    return chunks


class TTSDownloader:
    """Handles TTS download with detailed error tracking"""

    def __init__(self, headless: bool = True, timeout: int = 120000):
        self.headless = headless
        self.timeout = timeout
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def setup(self):
        """Initialize Playwright"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage"
            ]
        )
        self.context = await self.browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 800}
        )
        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.timeout)

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    async def download_chunk(
        self,
        chunk_index: int,
        text: str,
        output_dir: Path,
        total_chunks: int
    ) -> Tuple[int, bool, str, str]:
        """
        Download a single chunk with detailed error reporting
        Returns (chunk_index, success, filepath, error_message)
        """
        error_step = "initialization"

        try:
            # Step 1: Navigate
            error_step = "navigation"
            print(f"[{chunk_index + 1}/{total_chunks}] 🌐 Navigating to website...")
            await self.page.goto(TTS_URL, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            # Step 2: Handle consent
            error_step = "consent handling"
            print(f"[{chunk_index + 1}/{total_chunks}] 🍪 Handling consent...")
            try:
                consent_js = """
                () => {
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        const text = btn.textContent.toLowerCase();
                        if (text.includes('consent') || text.includes('agree') || text.includes('accept')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }
                """
                clicked = await self.page.evaluate(consent_js)
                if clicked:
                    print(f"[{chunk_index + 1}/{total_chunks}] ✅ Consent clicked")
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"[{chunk_index + 1}/{total_chunks}] ℹ️ No consent needed: {e}")

            # Step 3: Select locale
            error_step = "locale selection"
            print(f"[{chunk_index + 1}/{total_chunks}] 🌍 Selecting locale...")
            locale_js = """
            (locale) => {
                const selects = document.querySelectorAll('select');
                for (const select of selects) {
                    for (let i = 0; i < select.options.length; i++) {
                        if (select.options[i].text.includes(locale)) {
                            select.selectedIndex = i;
                            select.dispatchEvent(new Event('change'));
                            return select.options[i].text;
                        }
                    }
                }
                return null;
            }
            """
            locale_result = await self.page.evaluate(locale_js, LOCALE)
            if locale_result:
                print(f"[{chunk_index + 1}/{total_chunks}] ✅ Locale: {locale_result}")
                await asyncio.sleep(3)  # Wait for voices to load
            else:
                raise Exception(f"Could not select locale: {LOCALE}")

            # Step 4: Select voice
            error_step = "voice selection"
            print(f"[{chunk_index + 1}/{total_chunks}] 🎤 Selecting male voice...")
            voice_js = """
            () => {
                const selects = document.querySelectorAll('select');
                if (selects.length >= 2) {
                    const voiceSelect = selects[1];
                    for (let i = 0; i < voiceSelect.options.length; i++) {
                        const text = voiceSelect.options[i].text.toLowerCase();
                        if (text.includes('male') && !text.includes('female')) {
                            voiceSelect.selectedIndex = i;
                            voiceSelect.dispatchEvent(new Event('change'));
                            return voiceSelect.options[i].text;
                        }
                    }
                    // Fallback to first option
                    if (voiceSelect.options.length > 0) {
                        voiceSelect.selectedIndex = 0;
                        voiceSelect.dispatchEvent(new Event('change'));
                        return voiceSelect.options[0].text;
                    }
                }
                return null;
            }
            """
            voice_result = await self.page.evaluate(voice_js)
            if voice_result:
                print(f"[{chunk_index + 1}/{total_chunks}] ✅ Voice: {voice_result}")
                await asyncio.sleep(2)
            else:
                raise Exception("Could not select voice")

            # Step 5: Enter text
            error_step = "text entry"
            print(f"[{chunk_index + 1}/{total_chunks}] ⌨️ Entering text ({len(text)} chars)...")
            text_js = """
            (text) => {
                const textarea = document.querySelector('textarea');
                if (textarea) {
                    textarea.value = text;
                    textarea.dispatchEvent(new Event('input'));
                    return true;
                }
                return false;
            }
            """
            text_entered = await self.page.evaluate(text_js, text)
            if not text_entered:
                raise Exception("Could not enter text into textarea")
            print(f"[{chunk_index + 1}/{total_chunks}] ✅ Text entered")
            await asyncio.sleep(1)

            # Step 6: Click Play
            error_step = "play button click"
            print(f"[{chunk_index + 1}/{total_chunks}] 🔊 Clicking Play...")
            play_js = """
            () => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.id === 'quick-play' || btn.textContent.toLowerCase().includes('play')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }
            """
            play_clicked = await self.page.evaluate(play_js)
            if not play_clicked:
                raise Exception("Could not find or click Play button")
            print(f"[{chunk_index + 1}/{total_chunks}] ✅ Play clicked")

            # Step 7: Wait for generation (adaptive based on text length)
            error_step = "audio generation"
            wait_time = min(30, 10 + len(text) // 200)  # 10s base + 1s per 200 chars, max 30s
            print(f"[{chunk_index + 1}/{total_chunks}] ⏳ Waiting {wait_time}s for generation...")
            await asyncio.sleep(wait_time)

            # Step 8: Wait for Download button
            error_step = "download button ready"
            print(f"[{chunk_index + 1}/{total_chunks}] 🔍 Waiting for Download button...")

            # Try to wait for download button with detailed checking
            max_wait = 120  # 2 minutes max
            start_wait = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_wait
                if elapsed > max_wait:
                    raise Exception(f"Download button not ready after {max_wait}s")

                # Check if button exists and is enabled
                button_state = await self.page.evaluate("""
                () => {
                    const btn = document.querySelector('#download');
                    if (!btn) return 'not_found';
                    if (btn.disabled) return 'disabled';
                    return 'ready';
                }
                """)

                if button_state == 'ready':
                    print(f"[{chunk_index + 1}/{total_chunks}] ✅ Download button ready!")
                    break
                elif button_state == 'not_found':
                    raise Exception("Download button not found in page")
                else:  # disabled
                    if int(elapsed) % 10 == 0 and elapsed > 0:  # Log every 10s
                        print(f"[{chunk_index + 1}/{total_chunks}] ⏳ Still waiting... ({int(elapsed)}s)")
                    await asyncio.sleep(2)

            await asyncio.sleep(2)

            # Step 9: Download
            error_step = "file download"
            print(f"[{chunk_index + 1}/{total_chunks}] ⬇️ Starting download...")

            async with self.page.expect_download(timeout=60000) as download_info:
                await self.page.click('#download')

            download = await download_info.value

            # Step 10: Save file
            error_step = "file save"
            target_filename = f"segment_{chunk_index:05d}.mp3"
            target_path = output_dir / target_filename

            await download.save_as(str(target_path))
            await asyncio.sleep(1)

            # Step 11: Verify
            error_step = "file verification"
            if not target_path.exists():
                raise Exception("File not saved to disk")

            file_size = target_path.stat().st_size
            if file_size < 1000:
                raise Exception(f"File too small: {file_size} bytes")

            print(f"[{chunk_index + 1}/{total_chunks}] ✅ SUCCESS: {target_filename} ({file_size/1024:.1f} KB)")
            return (chunk_index, True, str(target_path), "")

        except Exception as e:
            error_msg = f"Failed at step '{error_step}': {str(e)}"
            print(f"[{chunk_index + 1}/{total_chunks}] ❌ {error_msg}")
            return (chunk_index, False, "", error_msg)


async def download_chunks_sequential(
    chunks: List[Tuple[int, str]],
    output_dir: Path,
    headless: bool = True
) -> List[Tuple[int, bool, str, str]]:
    """Download chunks one at a time (most reliable)"""
    results = []
    total_chunks = len(chunks)

    for chunk_index, text in chunks:
        downloader = TTSDownloader(headless=headless)

        try:
            await downloader.setup()
            result = await downloader.download_chunk(chunk_index, text, output_dir, total_chunks)
            results.append(result)
        except Exception as e:
            print(f"[{chunk_index + 1}/{total_chunks}] ❌ Fatal error: {e}")
            results.append((chunk_index, False, "", f"Fatal error: {e}"))
        finally:
            await downloader.cleanup()
            # Small delay between chunks
            await asyncio.sleep(2)

    return results


async def download_chunks_concurrent(
    chunks: List[Tuple[int, str]],
    output_dir: Path,
    max_concurrent: int = 3,
    headless: bool = True
) -> List[Tuple[int, bool, str, str]]:
    """Download chunks with limited concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(chunk_index, text):
        async with semaphore:
            downloader = TTSDownloader(headless=headless)
            try:
                await downloader.setup()
                result = await downloader.download_chunk(
                    chunk_index, text, output_dir, len(chunks)
                )
                return result
            except Exception as e:
                print(f"[{chunk_index + 1}/{len(chunks)}] ❌ Fatal error: {e}")
                return (chunk_index, False, "", f"Fatal error: {e}")
            finally:
                await downloader.cleanup()

    tasks = [download_with_semaphore(idx, txt) for idx, txt in chunks]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    parser = argparse.ArgumentParser(description="Parallel TTS Downloader")
    parser.add_argument("input_file", type=str, help="Input text file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--concurrent", type=int, default=1, help="Max concurrent downloads (1=sequential)")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Characters per chunk")
    parser.add_argument("--test", action="store_true", help="Test mode: only process first 2 chunks")
    parser.add_argument("--no-headless", action="store_true", help="Show browser windows")

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ File not found: {args.input_file}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_tts_output"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Read text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print("=" * 70)
    print("🎯 PARALLEL TTS DOWNLOADER")
    print("=" * 70)
    print(f"📄 Input: {input_path}")
    print(f"📝 Text: {len(text)} chars, {len(text.split())} words")
    print(f"📁 Output: {output_dir}")
    print(f"✂️ Chunk size: {args.chunk_size} chars")
    print(f"🔄 Concurrency: {args.concurrent}")
    print(f"👁️ Headless: {not args.no_headless}")
    if args.test:
        print("🧪 TEST MODE: Only processing first 2 chunks")
    print()

    # Chunk text
    print("✂️ Chunking text...")
    chunks = simple_chunk_text(text, args.chunk_size)
    print(f"📊 Created {len(chunks)} chunks")

    # Test mode: limit to first 2 chunks
    if args.test:
        chunks = chunks[:2]
        print(f"🧪 Test mode: Processing only {len(chunks)} chunks")

    print()

    # Download chunks
    start_time = datetime.now()

    if args.concurrent <= 1:
        print("🔄 Sequential download mode (most reliable)")
        results = await download_chunks_sequential(
            chunks, output_dir, headless=not args.no_headless
        )
    else:
        print(f"🔄 Concurrent download mode ({args.concurrent} parallel)")
        results = await download_chunks_concurrent(
            chunks, output_dir, args.concurrent, headless=not args.no_headless
        )

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(chunks)}")
    print(f"❌ Failed: {len(failed)}/{len(chunks)}")
    print(f"⏱️ Time: {elapsed:.1f}s")
    print(f"📁 Output: {output_dir}")

    if failed:
        print("\n❌ Failed chunks:")
        for chunk_idx, _, _, error in failed:
            print(f"   Chunk {chunk_idx}: {error}")

    print("=" * 70)

    return 0 if len(successful) == len(chunks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
