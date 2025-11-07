#!/usr/bin/env python3
"""
FIXED PARALLEL TTS DOWNLOADER - Actually works with detailed error messages
Simplified for testing, then can be enhanced with phoneme chunking
"""

import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import traceback

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("❌ Error: playwright not installed")
    print("Install: pip install playwright")
    print("Then run: python3 -m playwright install chromium")
    sys.exit(1)

# Configuration
TTS_WEBSITE = "https://www.text-to-speech.online/"
MAX_CONCURRENT = 5  # Start conservative
HEADLESS_MODE = True
LOCALE = "English (South Africa)"
PAGE_TIMEOUT = 45000  # 45 seconds
DOWNLOAD_TIMEOUT = 90000  # 90 seconds
GENERATION_WAIT = 15  # Wait 15s for audio generation


def simple_chunk_text(text: str, chunk_size: int = 4000) -> List[Tuple[int, str]]:
    """
    Simple word-boundary chunking for testing
    Returns list of (chunk_index, chunk_text) tuples
    """
    print(f"Chunking text (max {chunk_size} chars per chunk)...")

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_index, chunk_text))
            chunk_index += 1

            # Start new chunk
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Don't forget last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append((chunk_index, chunk_text))

    print(f"Created {len(chunks)} chunks")
    return chunks


async def download_chunk_with_detailed_errors(
    chunk_index: int,
    chunk_text: str,
    output_dir: Path,
    total_chunks: int,
    headless: bool = True,
    take_screenshots: bool = False
) -> Tuple[int, bool, str, str]:
    """
    Download a single chunk with DETAILED error reporting
    Returns (chunk_index, success, filepath, error_message)
    """
    error_log = []
    playwright = None
    browser = None

    try:
        print(f"\n{'='*70}")
        print(f"[{chunk_index + 1}/{total_chunks}] Starting download...")
        print(f"Text length: {len(chunk_text)} chars")
        print(f"{'='*70}")

        # Start Playwright
        error_log.append("Starting Playwright...")
        playwright = await async_playwright().start()

        # Launch browser
        error_log.append("Launching browser...")
        browser = await playwright.chromium.launch(
            headless=headless,
            args=['--no-sandbox', '--disable-blink-features=AutomationControlled']
        )

        # Create context
        error_log.append("Creating browser context...")
        context = await browser.new_context(
            accept_downloads=True,
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        page = await context.new_page()
        page.set_default_timeout(PAGE_TIMEOUT)

        # Navigate
        error_log.append(f"Navigating to {TTS_WEBSITE}...")
        print(f"   → Navigating to website...")
        await page.goto(TTS_WEBSITE, wait_until='domcontentloaded')
        await asyncio.sleep(2)

        if take_screenshots:
            screenshot_path = output_dir / f"debug_chunk_{chunk_index:03d}_1_loaded.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"   📸 Screenshot: {screenshot_path}")

        # Handle cookie consent
        error_log.append("Handling cookie consent...")
        print(f"   → Checking for cookie consent...")
        consent_handled = False
        consent_selectors = [
            'button:has-text("Consent")',
            '.fc-button.fc-cta-consent',
            'button:has-text("I agree")',
            'button:has-text("Accept")',
            'button:has-text("OK")'
        ]

        for selector in consent_selectors:
            try:
                button = page.locator(selector).first
                if await button.is_visible(timeout=2000):
                    await button.click()
                    print(f"   ✓ Clicked consent: {selector}")
                    await asyncio.sleep(2)
                    consent_handled = True
                    break
            except:
                continue

        if not consent_handled:
            print(f"   ℹ No consent dialog found (might be OK)")

        # Select locale
        error_log.append(f"Selecting locale: {LOCALE}...")
        print(f"   → Selecting locale: {LOCALE}...")
        try:
            # Find the first select element (locale)
            locale_select = page.locator('select').first
            await locale_select.wait_for(state='visible', timeout=10000)

            # Get available options
            options = await locale_select.locator('option').all_text_contents()
            print(f"   ℹ Available locales: {len(options)} options")

            # Try to select the locale
            await locale_select.select_option(label=LOCALE)
            print(f"   ✓ Selected locale: {LOCALE}")
            await asyncio.sleep(3)  # Wait for voice options to reload

        except Exception as e:
            error_msg = f"Locale selection failed: {e}"
            error_log.append(error_msg)
            print(f"   ⚠️ {error_msg}")
            # Continue anyway with default locale

        # Select voice (male)
        error_log.append("Selecting male voice...")
        print(f"   → Selecting male voice...")
        try:
            selects = await page.locator('select').all()
            print(f"   ℹ Found {len(selects)} select elements")

            if len(selects) >= 2:
                voice_select = selects[1]
                options = await voice_select.locator('option').all()
                print(f"   ℹ Voice options: {len(options)} available")

                # Find male voice
                male_voice_found = False
                for option in options:
                    text = await option.text_content()
                    text_lower = text.lower()
                    is_male = any(kw in text_lower for kw in ['male', 'guy', 'man', 'luke'])
                    is_female = 'female' in text_lower

                    if is_male and not is_female:
                        value = await option.get_attribute('value')
                        await voice_select.select_option(value=value)
                        print(f"   ✓ Selected voice: {text}")
                        male_voice_found = True
                        await asyncio.sleep(1)
                        break

                if not male_voice_found:
                    print(f"   ⚠️ No male voice found, using default")
            else:
                print(f"   ⚠️ Expected 2+ select elements, found {len(selects)}")

        except Exception as e:
            error_msg = f"Voice selection failed: {e}"
            error_log.append(error_msg)
            print(f"   ⚠️ {error_msg}")

        if take_screenshots:
            screenshot_path = output_dir / f"debug_chunk_{chunk_index:03d}_2_configured.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"   📸 Screenshot: {screenshot_path}")

        # Enter text
        error_log.append("Entering text into textarea...")
        print(f"   → Entering text ({len(chunk_text)} chars)...")
        try:
            textarea = page.locator('textarea').first
            await textarea.wait_for(state='visible', timeout=10000)
            await textarea.clear()
            await textarea.fill(chunk_text)
            print(f"   ✓ Text entered")
            await asyncio.sleep(1)
        except Exception as e:
            error_msg = f"Failed to enter text: {e}"
            error_log.append(error_msg)
            raise Exception(error_msg)

        # Click Play/Generate button
        error_log.append("Clicking Play button...")
        print(f"   → Clicking Play button to generate audio...")
        try:
            play_btn = page.locator('#quick-play')
            await play_btn.wait_for(state='visible', timeout=10000)

            # Check if button is disabled
            is_disabled = await play_btn.is_disabled()
            if is_disabled:
                raise Exception("Play button is disabled!")

            await play_btn.click()
            print(f"   ✓ Clicked Play button")

        except Exception as e:
            error_msg = f"Failed to click Play button: {e}"
            error_log.append(error_msg)
            raise Exception(error_msg)

        if take_screenshots:
            screenshot_path = output_dir / f"debug_chunk_{chunk_index:03d}_3_playing.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"   📸 Screenshot: {screenshot_path}")

        # Wait for audio generation
        error_log.append(f"Waiting {GENERATION_WAIT}s for audio generation...")
        print(f"   → Waiting {GENERATION_WAIT}s for audio generation...")
        await asyncio.sleep(GENERATION_WAIT)

        # Check download button status
        error_log.append("Checking download button status...")
        print(f"   → Checking download button status...")
        try:
            download_btn = page.locator('#download')

            # Wait for button to exist
            await download_btn.wait_for(state='attached', timeout=10000)
            print(f"   ✓ Download button exists")

            # Wait for button to be visible
            await download_btn.wait_for(state='visible', timeout=10000)
            print(f"   ✓ Download button visible")

            # Check if disabled
            is_disabled = await download_btn.is_disabled()
            print(f"   ℹ Download button disabled: {is_disabled}")

            if is_disabled:
                # Wait for it to be enabled (with detailed progress)
                print(f"   → Waiting for download button to be enabled...")
                for attempt in range(30):  # 30 attempts = 30 seconds
                    await asyncio.sleep(1)
                    is_disabled = await download_btn.is_disabled()
                    if not is_disabled:
                        print(f"   ✓ Download button enabled after {attempt + 1}s")
                        break
                    if attempt % 5 == 0:
                        print(f"   ⏳ Still waiting... ({attempt + 1}s)")

                # Final check
                is_disabled = await download_btn.is_disabled()
                if is_disabled:
                    # Take screenshot to see what's wrong
                    if take_screenshots:
                        screenshot_path = output_dir / f"debug_chunk_{chunk_index:03d}_4_stuck.png"
                        await page.screenshot(path=str(screenshot_path))
                        print(f"   📸 Screenshot: {screenshot_path}")

                    raise Exception(f"Download button still disabled after 30s wait!")

            print(f"   ✓ Download button is enabled")

        except Exception as e:
            error_msg = f"Download button check failed: {e}"
            error_log.append(error_msg)
            raise Exception(error_msg)

        if take_screenshots:
            screenshot_path = output_dir / f"debug_chunk_{chunk_index:03d}_5_ready.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"   📸 Screenshot: {screenshot_path}")

        # Download the file
        error_log.append("Initiating download...")
        print(f"   → Initiating download...")
        try:
            async with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as download_info:
                await download_btn.click()
                print(f"   ✓ Clicked download button")

            download = await download_info.value
            print(f"   ✓ Download started")

            # Save file
            target_filename = f"segment_{chunk_index:05d}.mp3"
            target_path = output_dir / target_filename

            await download.save_as(str(target_path))
            print(f"   ✓ File saved")
            await asyncio.sleep(1)

            # Verify
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size > 1000:
                    print(f"   ✅ SUCCESS: {target_filename} ({file_size/1024:.1f} KB)")
                    return (chunk_index, True, str(target_path), "")
                else:
                    error_msg = f"File too small: {file_size} bytes"
                    error_log.append(error_msg)
                    return (chunk_index, False, "", error_msg)
            else:
                error_msg = "File not found after download"
                error_log.append(error_msg)
                return (chunk_index, False, "", error_msg)

        except Exception as e:
            error_msg = f"Download failed: {e}"
            error_log.append(error_msg)
            raise Exception(error_msg)

    except Exception as e:
        error_summary = "\n".join([f"     {log}" for log in error_log])
        full_error = f"DETAILED ERROR LOG:\n{error_summary}\n   EXCEPTION: {str(e)}\n   TRACEBACK:\n{traceback.format_exc()}"
        print(f"\n❌ [{chunk_index + 1}/{total_chunks}] FAILED:")
        print(full_error)
        return (chunk_index, False, "", full_error)

    finally:
        # Cleanup
        try:
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
        except:
            pass


async def download_all_chunks_parallel(
    chunks: List[Tuple[int, str]],
    output_dir: Path,
    max_concurrent: int = 5,
    headless: bool = True,
    take_screenshots: bool = False
) -> List[Tuple[int, bool, str, str]]:
    """
    Download all chunks with concurrency control
    Returns list of (chunk_index, success, filepath, error) tuples
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total_chunks = len(chunks)

    async def download_with_semaphore(chunk_index, chunk_text):
        async with semaphore:
            return await download_chunk_with_detailed_errors(
                chunk_index, chunk_text, output_dir, total_chunks,
                headless, take_screenshots
            )

    print(f"\n🚀 Starting downloads: {total_chunks} chunks, {max_concurrent} concurrent")
    print("="*70)

    tasks = [
        download_with_semaphore(chunk_index, chunk_text)
        for chunk_index, chunk_text in chunks
    ]

    results = await asyncio.gather(*tasks)

    return results


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Parallel TTS Downloader with detailed errors")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent downloads (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=4000, help="Characters per chunk (default: 4000)")
    parser.add_argument("--no-headless", action="store_true", help="Show browser windows")
    parser.add_argument("--screenshots", action="store_true", help="Take debug screenshots")
    parser.add_argument("--test-one", action="store_true", help="Only test first chunk")

    args = parser.parse_args()

    print("="*70)
    print("🔧 FIXED PARALLEL TTS DOWNLOADER - WITH DETAILED ERRORS")
    print("="*70)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Input: {args.input_file}")
    print(f"🔄 Concurrency: {args.concurrent}")
    print(f"👁️ Headless: {not args.no_headless}")
    print(f"📸 Screenshots: {args.screenshots}")
    print()

    # Read input
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ File not found: {args.input_file}")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"📝 Read {len(text)} characters ({len(text.split())} words)")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_tts_output"

    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output: {output_dir.absolute()}")
    print()

    # Chunk text
    chunks = simple_chunk_text(text, args.chunk_size)

    # Test mode - only first chunk
    if args.test_one:
        print("\n⚠️ TEST MODE: Only processing first chunk")
        chunks = chunks[:1]

    print(f"📊 Processing {len(chunks)} chunks")
    print()

    # Download
    results = await download_all_chunks_parallel(
        chunks,
        output_dir,
        args.concurrent,
        headless=not args.no_headless,
        take_screenshots=args.screenshots
    )

    # Summary
    successful = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - successful

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\n❌ FAILED CHUNKS:")
        for chunk_index, success, _, error in results:
            if not success:
                print(f"\nChunk {chunk_index}:")
                print(f"{error[:500]}...")  # First 500 chars of error

    print("="*70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
