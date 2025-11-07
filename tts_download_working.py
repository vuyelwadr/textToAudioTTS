#!/usr/bin/env python3
"""
Working TTS Audio Downloader using Playwright
This script successfully downloads audio files from text-to-speech.online
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Configuration
TTS_WEBSITE = "https://www.text-to-speech.online/"
DOWNLOAD_TIMEOUT = 120000  # 2 minutes
PAGE_TIMEOUT = 60000  # 1 minute


async def download_tts_audio(text: str, output_path: str, headless: bool = True):
    """
    Download TTS audio using Playwright with explicit download control

    Args:
        text: Text to convert to speech
        output_path: Full path where audio file should be saved
        headless: Run browser in headless mode

    Returns:
        bool: True if download successful, False otherwise
    """
    print(f"🎙️  Starting TTS download...")
    print(f"📝 Text length: {len(text)} characters")
    print(f"💾 Target: {output_path}")

    async with async_playwright() as p:
        try:
            # Launch browser with download support
            print("🌐 Launching browser...")
            browser = await p.chromium.launch(headless=headless)

            # Create context with downloads enabled
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1280, 'height': 720}
            )

            # Create page
            page = await context.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)

            # Navigate to TTS website
            print(f"🔗 Navigating to {TTS_WEBSITE}...")
            await page.goto(TTS_WEBSITE, wait_until='networkidle')
            await asyncio.sleep(2)

            # Handle cookie consent if present
            try:
                cookie_button = page.locator('button:has-text("Accept"), button:has-text("I agree"), button:has-text("OK")')
                if await cookie_button.count() > 0:
                    print("🍪 Accepting cookies...")
                    await cookie_button.first.click()
                    await asyncio.sleep(1)
            except:
                pass

            # Select locale: English (South Africa)
            print("🌍 Setting locale to English (South Africa)...")
            try:
                locale_select = page.locator('select#locale, select[name="locale"]')
                await locale_select.select_option(label="English (South Africa)")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"⚠️  Warning: Could not set locale: {e}")
                print("    Continuing with default locale...")

            # Select voice (Male - Microsoft Guy Online)
            print("🗣️  Setting voice...")
            try:
                voice_select = page.locator('select#voice, select[name="voice"]')
                # Try to select Microsoft Guy, fallback to first male voice
                try:
                    await voice_select.select_option(label="Microsoft Guy Online (Natural)")
                except:
                    # Fallback: select any option containing "Male" or "Guy"
                    options = await voice_select.locator('option').all_text_contents()
                    for i, option in enumerate(options):
                        if any(keyword in option.lower() for keyword in ['male', 'guy', 'man']):
                            await voice_select.select_option(index=i)
                            break
                await asyncio.sleep(1)
            except Exception as e:
                print(f"⚠️  Warning: Could not set voice: {e}")
                print("    Continuing with default voice...")

            # Input text
            print("⌨️  Entering text...")
            textarea = page.locator('textarea#text, textarea[name="text"], textarea.form-control')
            await textarea.fill(text)
            await asyncio.sleep(1)

            # Click generate/play button first to generate audio
            print("🎵 Generating audio...")
            try:
                generate_button = page.locator('button:has-text("Generate"), button:has-text("Play"), button#btnSpeak')
                await generate_button.first.click()
                await asyncio.sleep(5)  # Wait for audio generation
            except Exception as e:
                print(f"⚠️  Warning: Could not find generate button: {e}")

            # Now attempt to download
            print("⬇️  Initiating download...")

            # Set up download promise BEFORE clicking download button
            download_promise = page.wait_for_event('download', timeout=DOWNLOAD_TIMEOUT)

            # Click download button
            try:
                download_button = page.locator(
                    'button:has-text("Download"), a:has-text("Download"), '
                    'button[title*="Download"], a[download], '
                    'button.download, a.download'
                )
                await download_button.first.click()
            except Exception as e:
                print(f"❌ Could not find download button: {e}")
                await browser.close()
                return False

            # Wait for download to start
            print("⏳ Waiting for download...")
            download = await download_promise

            # Save to specified path
            print(f"💾 Saving to {output_path}...")
            await download.save_as(output_path)

            # Verify file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    print(f"✅ SUCCESS! Downloaded {file_size:,} bytes")
                    print(f"📁 File saved to: {output_path}")
                    await browser.close()
                    return True
                else:
                    print(f"❌ ERROR: File is empty")
                    await browser.close()
                    return False
            else:
                print(f"❌ ERROR: File not found at {output_path}")
                await browser.close()
                return False

        except PlaywrightTimeout as e:
            print(f"❌ TIMEOUT ERROR: {e}")
            print("    The download took too long. The website might be slow or blocked.")
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            try:
                await browser.close()
            except:
                pass


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python tts_download_working.py <text_file> [output_dir]")
        print("Example: python tts_download_working.py test.txt ./output")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./tts_output"

    # Validate input file
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Read text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        print("❌ ERROR: Input file is empty")
        sys.exit(1)

    # Limit text length for testing (TTS websites have limits)
    if len(text) > 5000:
        print(f"⚠️  Text is {len(text)} chars. Truncating to first 5000 for testing...")
        text = text[:5000]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tts_audio_{timestamp}.mp3")

    print("=" * 70)
    print("🎙️  TTS Audio Downloader - Working Version")
    print("=" * 70)

    # Run download
    success = asyncio.run(download_tts_audio(text, output_path, headless=True))

    print("=" * 70)
    if success:
        print("🎉 DOWNLOAD SUCCESSFUL!")
        print(f"📁 Audio file: {output_path}")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ DOWNLOAD FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
