#!/usr/bin/env python3
"""
PRODUCTION TTS DOWNLOAD SCRIPT
- Robust error handling
- Headless and headed modes
- Progress tracking
- Multiple chunk support
- Session reporting
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
except ImportError as e:
    print(f"❌ Missing playwright: {e}")
    print("Install with: pip install playwright && playwright install chromium")
    sys.exit(1)

# Configuration
DEFAULT_LOCALE = "English (South Africa)"
DEFAULT_VOICE_FILTER = "male"  # Will select first male voice
TTS_URL = "https://www.text-to-speech.online/"
CHUNK_SIZE = 5000  # Characters per chunk
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class TTSDownloader:
    """Handles TTS generation and download using Playwright"""

    def __init__(self, output_dir: Path, headless: bool = True):
        self.output_dir = output_dir
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.stats = {
            "total_chunks": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "start_time": None,
            "end_time": None
        }

    async def setup(self):
        """Initialize Playwright and browser"""
        print("🚀 Setting up Playwright...")
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
            viewport={"width": 1280, "height": 720}
        )

        self.page = await self.context.new_page()
        print("✅ Playwright setup complete")

    async def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    async def handle_consent(self):
        """Handle cookie consent popup"""
        consent_js = """
        () => {
            const consentTexts = ['I agree', 'Consent', 'Accept', 'OK'];
            const buttons = document.querySelectorAll('button');
            for (const button of buttons) {
                if (consentTexts.some(text => button.textContent.includes(text))) {
                    button.click();
                    return true;
                }
            }
            return false;
        }
        """

        try:
            consent_result = await self.page.evaluate(consent_js)
            if consent_result:
                print("✅ Consent accepted")
                await asyncio.sleep(2)

            # Remove overlays
            overlay_js = """
            () => {
                const overlays = document.querySelectorAll('[class*="overlay"], [class*="modal"], [class*="consent"]');
                for (const overlay of overlays) {
                    overlay.style.display = 'none';
                    overlay.remove();
                }
                return true;
            }
            """
            await self.page.evaluate(overlay_js)
        except Exception as e:
            print(f"ℹ️ Consent handling: {e}")

    async def select_locale(self, locale: str = DEFAULT_LOCALE):
        """Select locale from dropdown"""
        locale_js = """
        (locale) => {
            const selects = document.querySelectorAll('select');
            for (const select of selects) {
                const options = select.options;
                for (let i = 0; i < options.length; i++) {
                    if (options[i].text.includes(locale)) {
                        select.selectedIndex = i;
                        select.dispatchEvent(new Event('change'));
                        return options[i].text;
                    }
                }
            }
            return null;
        }
        """

        try:
            result = await self.page.evaluate(locale_js, locale)
            if result:
                print(f"✅ Locale: {result}")
                await asyncio.sleep(2)
                return True
            else:
                print(f"⚠️ Could not find locale: {locale}")
                return False
        except Exception as e:
            print(f"❌ Locale error: {e}")
            return False

    async def select_voice(self, voice_filter: str = DEFAULT_VOICE_FILTER):
        """Select voice from dropdown"""
        voice_js = """
        (filter) => {
            const selects = document.querySelectorAll('select');
            if (selects.length >= 2) {
                const voiceSelect = selects[1];
                const options = voiceSelect.options;

                // Try to find voice matching filter
                for (let i = 0; i < options.length; i++) {
                    const text = options[i].text.toLowerCase();
                    if (text.includes(filter.toLowerCase())) {
                        voiceSelect.selectedIndex = i;
                        voiceSelect.dispatchEvent(new Event('change'));
                        return options[i].text;
                    }
                }

                // Fallback to first option
                if (options.length > 0) {
                    voiceSelect.selectedIndex = 0;
                    voiceSelect.dispatchEvent(new Event('change'));
                    return options[0].text;
                }
            }
            return null;
        }
        """

        try:
            result = await self.page.evaluate(voice_js, voice_filter)
            if result:
                print(f"✅ Voice: {result}")
                await asyncio.sleep(2)
                return True
            else:
                print(f"⚠️ Could not select voice")
                return False
        except Exception as e:
            print(f"❌ Voice error: {e}")
            return False

    async def enter_text(self, text: str):
        """Enter text into textarea"""
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

        try:
            result = await self.page.evaluate(text_js, text)
            if result:
                print(f"✅ Text entered ({len(text)} chars)")
                await asyncio.sleep(1)
                return True
            else:
                print("❌ Could not enter text")
                return False
        except Exception as e:
            print(f"❌ Text error: {e}")
            return False

    async def click_play(self):
        """Click the Play/Quick Play button"""
        play_js = """
        () => {
            const buttons = document.querySelectorAll('button');
            for (const button of buttons) {
                const text = button.textContent.toLowerCase();
                if (text.includes('play') || button.id === 'quick-play') {
                    button.click();
                    return button.textContent;
                }
            }
            return null;
        }
        """

        try:
            result = await self.page.evaluate(play_js)
            if result:
                print(f"✅ Clicked: {result}")
                await asyncio.sleep(12)  # Wait for TTS processing
                return True
            else:
                print("❌ Could not find Play button")
                return False
        except Exception as e:
            print(f"❌ Play error: {e}")
            return False

    async def download_audio(self, chunk_index: int):
        """Download the generated audio file"""
        try:
            # Wait for download button
            await self.page.wait_for_selector(
                'button:has-text("Download")',
                timeout=30000,
                state='visible'
            )
            print("✅ Download button ready")
            await asyncio.sleep(2)

            # Set up download promise BEFORE clicking
            async with self.page.expect_download(timeout=60000) as download_info:
                # Click download button
                await self.page.click('button:has-text("Download")')
                print("⬇️ Download started...")

            # Get download object
            download = await download_info.value
            suggested_name = download.suggested_filename
            print(f"📄 Captured: {suggested_name}")

            # Save to specific path
            target_path = self.output_dir / f"chunk_{chunk_index:04d}.mp3"
            await download.save_as(str(target_path))

            # Verify file
            await asyncio.sleep(2)
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size > 1000:
                    print(f"✅ Downloaded: {target_path.name} ({file_size:,} bytes)")
                    self.stats["successful_downloads"] += 1
                    return True
                else:
                    print(f"⚠️ File too small: {file_size} bytes")
                    self.stats["failed_downloads"] += 1
                    return False
            else:
                print("❌ File not found after download")
                self.stats["failed_downloads"] += 1
                return False

        except PlaywrightTimeoutError:
            print("❌ Download timeout")
            self.stats["failed_downloads"] += 1
            return False
        except Exception as e:
            print(f"❌ Download error: {e}")
            self.stats["failed_downloads"] += 1
            return False

    async def process_chunk(self, text: str, chunk_index: int, retry_count: int = 0):
        """Process a single text chunk"""
        print(f"\n{'='*70}")
        print(f"📝 Chunk {chunk_index + 1} (attempt {retry_count + 1}/{MAX_RETRIES})")
        print(f"{'='*70}")

        try:
            # Navigate to fresh page
            print("🌐 Loading TTS website...")
            await self.page.goto(TTS_URL, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            # Handle consent
            await self.handle_consent()

            # Select locale
            if not await self.select_locale():
                raise Exception("Locale selection failed")

            # Select voice
            if not await self.select_voice():
                raise Exception("Voice selection failed")

            # Enter text
            if not await self.enter_text(text):
                raise Exception("Text entry failed")

            # Click play
            if not await self.click_play():
                raise Exception("Play button failed")

            # Download
            if await self.download_audio(chunk_index):
                return True
            else:
                raise Exception("Download failed")

        except Exception as e:
            print(f"❌ Chunk {chunk_index + 1} error: {e}")

            # Retry logic
            if retry_count < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (retry_count + 1)
                print(f"🔄 Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self.process_chunk(text, chunk_index, retry_count + 1)
            else:
                print(f"❌ Chunk {chunk_index + 1} failed after {MAX_RETRIES} attempts")
                return False

    async def process_file(self, text_file: Path, start_chunk: int = 0, end_chunk: int = None):
        """Process entire text file"""
        print(f"📖 Reading: {text_file}")

        with open(text_file, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split into chunks
        chunks = []
        for i in range(0, len(full_text), CHUNK_SIZE):
            chunks.append(full_text[i:i + CHUNK_SIZE])

        # Apply range
        if end_chunk is None:
            end_chunk = len(chunks)
        chunks_to_process = chunks[start_chunk:end_chunk]

        self.stats["total_chunks"] = len(chunks_to_process)
        self.stats["start_time"] = datetime.now().isoformat()

        print(f"📊 Total chunks: {len(chunks)} (processing {start_chunk} to {end_chunk - 1})")

        # Process chunks
        for i, chunk in enumerate(chunks_to_process, start=start_chunk):
            await self.process_chunk(chunk, i)

        self.stats["end_time"] = datetime.now().isoformat()

        # Save report
        report_file = self.output_dir / f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"\n{'='*70}")
        print(f"📊 SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successful: {self.stats['successful_downloads']}")
        print(f"❌ Failed: {self.stats['failed_downloads']}")
        print(f"📁 Output: {self.output_dir}")
        print(f"📄 Report: {report_file.name}")
        print(f"{'='*70}")


async def main():
    parser = argparse.ArgumentParser(description="TTS Online Download Script")
    parser.add_argument("input_file", type=Path, help="Input text file")
    parser.add_argument("-o", "--output", type=Path, default=Path("tts_output"), help="Output directory")
    parser.add_argument("--start-chunk", type=int, default=0, help="Start chunk index")
    parser.add_argument("--end-chunk", type=int, default=None, help="End chunk index")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")

    args = parser.parse_args()

    # Validate input
    if not args.input_file.exists():
        print(f"❌ File not found: {args.input_file}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run downloader
    downloader = TTSDownloader(args.output, headless=not args.no_headless)

    try:
        await downloader.setup()
        await downloader.process_file(args.input_file, args.start_chunk, args.end_chunk)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await downloader.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
