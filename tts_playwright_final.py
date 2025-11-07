#!/usr/bin/env python3
"""
TTS Audio Downloader - Playwright Version (WORKING)
Successfully downloads audio files from text-to-speech.online with explicit path control

This script solves the download path problem using Playwright's download API
"""

import asyncio
import os
import sys
import time
import argparse
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Configuration
TTS_WEBSITE = "https://www.text-to-speech.online/"
DOWNLOAD_TIMEOUT = 120000  # 2 minutes
PAGE_TIMEOUT = 60000  # 1 minute
DEFAULT_CHUNK_SIZE = 5000


async def download_tts_audio(text: str, output_path: str, headless: bool = True, locale: str = "English (South Africa)", voice: str = None):
    """
    Download TTS audio using Playwright with explicit download control

    Args:
        text: Text to convert to speech
        output_path: Full path where audio file should be saved
        headless: Run browser in headless mode
        locale: TTS locale (default: English South Africa)
        voice: TTS voice (default: Male voice)

    Returns:
        bool: True if download successful, False otherwise
    """
    print(f"🎙️  Starting TTS download...")
    print(f"📝 Text length: {len(text)} characters")
    print(f"🌍 Locale: {locale}")
    print(f"💾 Target: {output_path}")

    async with async_playwright() as p:
        browser = None
        try:
            # Launch browser with download support
            print("🌐 Launching browser...")
            browser = await p.chromium.launch(
                headless=headless,
                args=['--disable-blink-features=AutomationControlled']
            )

            # Create context with downloads enabled
            context = await browser.new_context(
                accept_downloads=True,
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            # Create page
            page = await context.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)

            # Navigate to TTS website
            print(f"🔗 Navigating to {TTS_WEBSITE}...")
            await page.goto(TTS_WEBSITE, wait_until='domcontentloaded')
            await asyncio.sleep(3)

            # Handle cookie consent if present
            try:
                print("🍪 Checking for cookie consent...")
                cookie_selectors = [
                    'button:has-text("Accept")',
                    'button:has-text("I agree")',
                    'button:has-text("OK")',
                    'button:has-text("Got it")',
                    'button.cc-allow',
                    'button#accept-cookies'
                ]
                for selector in cookie_selectors:
                    try:
                        button = page.locator(selector).first
                        if await button.is_visible(timeout=2000):
                            await button.click()
                            print("   Accepted cookies")
                            await asyncio.sleep(1)
                            break
                    except:
                        continue
            except Exception as e:
                print(f"   No cookie consent needed")

            # Select locale
            print(f"🌍 Setting locale to {locale}...")
            try:
                locale_select = page.locator('select#locale, select[name="locale"], select.form-select').first
                await locale_select.wait_for(state='visible', timeout=10000)
                await locale_select.select_option(label=locale)
                await asyncio.sleep(2)
                print("   Locale set successfully")
            except Exception as e:
                print(f"⚠️  Warning: Could not set locale: {e}")
                print("   Continuing with default locale...")

            # Select voice if specified
            if voice:
                print(f"🗣️  Setting voice to {voice}...")
                try:
                    voice_select = page.locator('select#voice, select[name="voice"], select.form-select').nth(1)
                    await voice_select.wait_for(state='visible', timeout=10000)
                    await voice_select.select_option(label=voice)
                    await asyncio.sleep(1)
                    print("   Voice set successfully")
                except Exception as e:
                    print(f"⚠️  Warning: Could not set voice: {e}")

            # Input text
            print("⌨️  Entering text...")
            textarea_selectors = [
                'textarea#text',
                'textarea[name="text"]',
                'textarea.form-control',
                'textarea'
            ]
            textarea = None
            for selector in textarea_selectors:
                try:
                    textarea = page.locator(selector).first
                    if await textarea.is_visible(timeout=2000):
                        break
                except:
                    continue

            if not textarea:
                raise Exception("Could not find text input area")

            await textarea.fill(text)
            await asyncio.sleep(1)
            print("   Text entered successfully")

            # Click generate/speak button
            print("🎵 Generating audio...")
            generate_selectors = [
                'button:has-text("Generate")',
                'button:has-text("Speak")',
                'button:has-text("Play")',
                'button#btnSpeak',
                'button.btn-primary'
            ]
            for selector in generate_selectors:
                try:
                    button = page.locator(selector).first
                    if await button.is_visible(timeout=2000):
                        await button.click()
                        print("   Audio generation started")
                        await asyncio.sleep(8)  # Wait for generation
                        break
                except:
                    continue

            # Now download the audio
            print("⬇️  Initiating download...")

            # Set up download promise BEFORE clicking
            download_promise = page.wait_for_event('download', timeout=DOWNLOAD_TIMEOUT)

            # Try multiple download button selectors
            download_selectors = [
                'button:has-text("Download")',
                'a:has-text("Download")',
                'button[title*="download" i]',
                'a[download]',
                'button.download',
                'a.download',
                'i.fa-download'
            ]

            download_clicked = False
            for selector in download_selectors:
                try:
                    button = page.locator(selector).first
                    if await button.is_visible(timeout=2000):
                        await button.click()
                        print(f"   Clicked download button: {selector}")
                        download_clicked = True
                        break
                except:
                    continue

            if not download_clicked:
                print("❌ Could not find download button")
                await browser.close()
                return False

            # Wait for download to start
            print("⏳ Waiting for download to start...")
            download = await download_promise
            print("   Download started!")

            # Save to specified path
            print(f"💾 Saving file...")
            await download.save_as(output_path)

            # Verify file
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    print(f"✅ SUCCESS! Downloaded {file_size:,} bytes")
                    print(f"📁 File saved to: {output_path}")
                    return True
                else:
                    print(f"❌ ERROR: File is too small ({file_size} bytes)")
                    return False
            else:
                print(f"❌ ERROR: File not created")
                return False

        except PlaywrightTimeout as e:
            print(f"❌ TIMEOUT ERROR: Operation took too long")
            print(f"   Details: {str(e)[:200]}")
            return False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if browser:
                try:
                    await browser.close()
                except:
                    pass


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list:
    """Split text into chunks"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='TTS Audio Downloader using Playwright')
    parser.add_argument('input_file', help='Path to input text file')
    parser.add_argument('-o', '--output', default='./tts_output', help='Output directory')
    parser.add_argument('--start-chunk', type=int, default=0, help='Start chunk index')
    parser.add_argument('--end-chunk', type=int, default=None, help='End chunk index')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help='Max characters per chunk')
    parser.add_argument('--no-headless', action='store_true', help='Show browser window')
    parser.add_argument('--locale', default='English (South Africa)', help='TTS locale')
    parser.add_argument('--voice', default=None, help='TTS voice name')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"❌ ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    # Read text
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        print("❌ ERROR: Input file is empty")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Split into chunks
    chunks = chunk_text(text, args.chunk_size)
    total_chunks = len(chunks)

    # Apply chunk range
    start_idx = args.start_chunk
    end_idx = args.end_chunk if args.end_chunk else total_chunks
    end_idx = min(end_idx, total_chunks)

    print("=" * 70)
    print("🎙️  TTS Audio Downloader - Playwright Version")
    print("=" * 70)
    print(f"📄 Input: {args.input_file}")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"🎯 Processing: {start_idx} to {end_idx-1}")
    print(f"📁 Output: {args.output}")
    print("=" * 70)

    successful = 0
    failed = 0

    for i in range(start_idx, end_idx):
        print(f"\n{'='*70}")
        print(f"Processing chunk {i+1}/{total_chunks}")
        print(f"{'='*70}")

        chunk_text_content = chunks[i]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output, f"chunk_{i:04d}_{timestamp}.mp3")

        success = asyncio.run(download_tts_audio(
            chunk_text_content,
            output_path,
            headless=not args.no_headless,
            locale=args.locale,
            voice=args.voice
        ))

        if success:
            successful += 1
        else:
            failed += 1
            print(f"⚠️  Failed to download chunk {i}")

        # Delay between chunks
        if i < end_idx - 1:
            print("\n⏸️  Pausing 3 seconds before next chunk...")
            time.sleep(3)

    # Final report
    print("\n" + "=" * 70)
    print("📊 FINAL REPORT")
    print("=" * 70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {args.output}")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
