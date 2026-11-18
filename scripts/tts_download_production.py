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
import base64
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
DEFAULT_VOICE_NAME = "LukeNeural(Male)"
DEFAULT_VOICE_FILTER = "male"  # Fallback only when the exact voice is unavailable
TTS_URL = "https://www.text-to-speech.online/"
CHUNK_SIZE = 5000  # Characters per chunk
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DOWNLOAD_CAPTURE_TIMEOUT_MS = 180000


class TTSDownloader:
    """Handles TTS generation and download using Playwright"""

    def __init__(
        self,
        output_dir: Path,
        headless: bool = True,
        locale: str = DEFAULT_LOCALE,
        voice_name: str = DEFAULT_VOICE_NAME,
        voice_filter: str = DEFAULT_VOICE_FILTER,
        require_exact_voice: bool = False,
    ):
        self.output_dir = output_dir
        self.headless = headless
        self.locale = locale
        self.voice_name = voice_name
        self.voice_filter = voice_filter
        self.require_exact_voice = require_exact_voice
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

        await self.context.add_init_script(
            """
            window.__copilotTtsCapture = { blob: null, dataUrl: null, filename: null, createdAt: null };

            (() => {
                const originalCreateObjectURL = URL.createObjectURL.bind(URL);
                URL.createObjectURL = function(blob) {
                    const normalizedBlob = blob instanceof Blob ? blob : new Blob([blob], { type: 'audio/mpeg' });
                    window.__copilotTtsCapture = {
                        blob: normalizedBlob,
                        dataUrl: null,
                        filename: window.__copilotTtsCapture?.filename || null,
                        createdAt: Date.now()
                    };

                    const reader = new FileReader();
                    reader.onloadend = () => {
                        window.__copilotTtsCapture = {
                            ...window.__copilotTtsCapture,
                            dataUrl: reader.result
                        };
                    };
                    reader.readAsDataURL(normalizedBlob);

                    return originalCreateObjectURL(blob);
                };

                const originalClick = HTMLAnchorElement.prototype.click;
                HTMLAnchorElement.prototype.click = function() {
                    if (this.download && this.href && this.href.startsWith('blob:')) {
                        window.__copilotTtsCapture = {
                            blob: window.__copilotTtsCapture?.blob || null,
                            dataUrl: window.__copilotTtsCapture?.dataUrl || null,
                            filename: this.download,
                            createdAt: window.__copilotTtsCapture?.createdAt || Date.now()
                        };
                        return;
                    }
                    return originalClick.call(this);
                };
            })();
            """
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
                await self.page.wait_for_function(
                    """
                    () => {
                        const selects = document.querySelectorAll('select');
                        return selects.length >= 2 && selects[1].options.length > 0;
                    }
                    """,
                    timeout=15000,
                )
                await asyncio.sleep(2)
                return True
            else:
                print(f"⚠️ Could not find locale: {locale}")
                return False
        except Exception as e:
            print(f"❌ Locale error: {e}")
            return False

    async def select_voice(self, locale: str = DEFAULT_LOCALE, voice_name: str = DEFAULT_VOICE_NAME, voice_filter: str = DEFAULT_VOICE_FILTER):
        """Select voice from dropdown"""
        voice_js = """
        ({ preferredVoice, filter, requireExactVoice }) => {
            const selects = document.querySelectorAll('select');
            if (selects.length >= 2) {
                const voiceSelect = selects[1];
                const options = Array.from(voiceSelect.options).map((option, index) => ({
                    index,
                    text: option.text.trim()
                }));

                if (preferredVoice) {
                    const exactOption = options.find((option) => option.text === preferredVoice);
                    if (exactOption) {
                        voiceSelect.selectedIndex = exactOption.index;
                        voiceSelect.dispatchEvent(new Event('change', { bubbles: true }));
                        return {
                            selected: exactOption.text,
                            mode: 'exact',
                            available: options.map((option) => option.text)
                        };
                    }
                }

                if (preferredVoice && requireExactVoice) {
                    return {
                        selected: null,
                        mode: 'missing-exact',
                        available: options.map((option) => option.text)
                    };
                }

                const normalizedFilter = (filter || '').toLowerCase();
                const filteredOption = options.find((option) => {
                    const text = option.text.toLowerCase();
                    return normalizedFilter && text.includes(normalizedFilter) && !text.includes('female');
                });
                if (filteredOption) {
                    voiceSelect.selectedIndex = filteredOption.index;
                    voiceSelect.dispatchEvent(new Event('change', { bubbles: true }));
                    return {
                        selected: filteredOption.text,
                        mode: 'filtered',
                        available: options.map((option) => option.text)
                    };
                }

                // Fallback to first option
                if (options.length > 0) {
                    voiceSelect.selectedIndex = 0;
                    voiceSelect.dispatchEvent(new Event('change', { bubbles: true }));
                    return {
                        selected: options[0].text,
                        mode: 'fallback',
                        available: options.map((option) => option.text)
                    };
                }
            }
            return { selected: null, mode: null, available: [] };
        }
        """

        try:
            result = await self.page.evaluate(
                voice_js,
                {
                    "preferredVoice": voice_name,
                    "filter": voice_filter,
                    "requireExactVoice": self.require_exact_voice,
                },
            )
            if result and result.get("selected"):
                print(f"✅ Voice: {result['selected']} ({result['mode']})")
                await asyncio.sleep(2)
                return True
            elif result and result.get("mode") == "missing-exact":
                print(f"⚠️ Exact voice not available: {voice_name}")
                return False
            else:
                print(f"⚠️ Could not select voice for locale: {locale}")
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
            await self.page.wait_for_function(
                """
                () => {
                    const downloadButton = document.querySelector('#download')
                        || Array.from(document.querySelectorAll('button')).find((button) =>
                            button.textContent && button.textContent.toLowerCase().includes('download')
                        );

                    if (!downloadButton) {
                        return false;
                    }

                    const label = (downloadButton.textContent || '').toLowerCase();
                    return !downloadButton.disabled && !label.includes('downloading');
                }
                """,
                timeout=DOWNLOAD_CAPTURE_TIMEOUT_MS,
            )
            print("✅ Download button enabled")
            await asyncio.sleep(2)

            await self.page.evaluate(
                """
                () => {
                    window.__copilotTtsCapture = { blob: null, dataUrl: null, filename: null, createdAt: null };

                    if (!window.__copilotTtsCaptureWrapped) {
                        const originalCreateObjectURL = URL.createObjectURL.bind(URL);
                        URL.createObjectURL = function(blob) {
                            const normalizedBlob = blob instanceof Blob ? blob : new Blob([blob], { type: 'audio/mpeg' });
                            window.__copilotTtsCapture = {
                                blob: normalizedBlob,
                                dataUrl: null,
                                filename: window.__copilotTtsCapture?.filename || null,
                                createdAt: Date.now()
                            };

                            const reader = new FileReader();
                            reader.onloadend = () => {
                                window.__copilotTtsCapture = {
                                    ...window.__copilotTtsCapture,
                                    dataUrl: reader.result
                                };
                            };
                            reader.readAsDataURL(normalizedBlob);

                            return originalCreateObjectURL(blob);
                        };

                        const originalAnchorClick = HTMLAnchorElement.prototype.click;
                        HTMLAnchorElement.prototype.click = function() {
                            if (this.download && this.href && (this.href.startsWith('blob:') || this.href.startsWith('data:'))) {
                                window.__copilotTtsCapture = {
                                    blob: window.__copilotTtsCapture?.blob || null,
                                    dataUrl: this.href.startsWith('data:') ? this.href : window.__copilotTtsCapture?.dataUrl || null,
                                    filename: this.download,
                                    createdAt: window.__copilotTtsCapture?.createdAt || Date.now()
                                };
                                return;
                            }
                            return originalAnchorClick.call(this);
                        };

                        window.__copilotTtsCaptureWrapped = true;
                    }
                }
                """
            )

            target_path = self.output_dir / f"chunk_{chunk_index:04d}.mp3"
            await self.page.evaluate(
                """
                () => {
                    const downloadButton = document.querySelector('#download');
                    if (downloadButton) {
                        downloadButton.click();
                    }
                }
                """
            )
            print("⬇️ Download click issued, waiting for in-page audio capture...")

            capture = await self.page.evaluate(
                """
                async (timeoutMs) => {
                    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
                    const start = Date.now();

                    while (Date.now() - start < timeoutMs) {
                        const state = window.__copilotTtsCapture || {};
                        if (state.dataUrl) {
                            const filename = state.filename || 'speech.mp3';
                            const dataUrl = state.dataUrl;
                            window.__copilotTtsCapture = { blob: null, dataUrl: null, filename: null, createdAt: null };
                            return { filename, dataUrl };
                        }
                        await sleep(500);
                    }

                    return null;
                }
                """,
                DOWNLOAD_CAPTURE_TIMEOUT_MS,
            )

            if not capture or not capture.get("dataUrl"):
                print("❌ Download capture timed out")
                self.stats["failed_downloads"] += 1
                return False

            print(f"📄 Captured: {capture.get('filename', 'speech.mp3')}")

            data_url = capture["dataUrl"]
            if "," not in data_url:
                print("❌ Invalid blob payload returned from page")
                self.stats["failed_downloads"] += 1
                return False

            _, encoded_audio = data_url.split(",", 1)
            audio_bytes = base64.b64decode(encoded_audio)

            # Save to specific path
            target_path.write_bytes(audio_bytes)

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
            if not await self.select_locale(self.locale):
                raise Exception("Locale selection failed")

            # Select voice
            if not await self.select_voice(locale=self.locale, voice_name=self.voice_name, voice_filter=self.voice_filter):
                raise Exception("Voice selection failed")

            # Enter text
            if not await self.enter_text(text):
                raise Exception("Text entry failed")

            # The site behaves more reliably when playback is primed before download.
            if not await self.click_play():
                raise Exception("Play button failed")

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
    parser.add_argument("--locale", default=DEFAULT_LOCALE, help="Locale label to select")
    parser.add_argument("--voice-name", default=DEFAULT_VOICE_NAME, help="Exact voice label to prefer")
    parser.add_argument("--voice-filter", default=DEFAULT_VOICE_FILTER, help="Fallback voice filter if the exact label is unavailable")
    parser.add_argument("--require-exact-voice", action="store_true", help="Fail if the exact voice label is unavailable")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")

    args = parser.parse_args()

    # Validate input
    if not args.input_file.exists():
        print(f"❌ File not found: {args.input_file}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run downloader
    downloader = TTSDownloader(
        args.output,
        headless=not args.no_headless,
        locale=args.locale,
        voice_name=args.voice_name,
        voice_filter=args.voice_filter,
        require_exact_voice=args.require_exact_voice,
    )

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
