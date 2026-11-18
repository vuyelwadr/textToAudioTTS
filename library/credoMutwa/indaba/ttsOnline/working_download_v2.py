#!/usr/bin/env python3
"""
WORKING TTS DOWNLOAD - Clean implementation that actually downloads
Target: debug_download2/
Language: English (South Africa)
Voice: Male
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Error: playwright not installed")
    print("Install: pip install playwright")
    print("Then run: python3 -m playwright install chromium")
    sys.exit(1)

async def download_tts_audio():
    """Download audio from text-to-speech.online"""

    print("=" * 70)
    print("🎯 TTS ONLINE AUDIO DOWNLOADER")
    print("=" * 70)

    # Setup output directory
    output_dir = Path("debug_download2")
    output_dir.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_dir.absolute()}")

    # Test text
    test_text = "Hello world. This is a working test of audio download."
    print(f"📝 Test text: {test_text}")
    print()

    # Start Playwright
    print("🚀 Starting Playwright...")
    playwright = await async_playwright().start()

    try:
        # Launch browser (headless mode - no popup)
        print("🌐 Launching browser...")
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled"
            ]
        )

        # Create context with downloads enabled
        context = await browser.new_context(
            accept_downloads=True,
            viewport={"width": 1280, "height": 800}
        )

        page = await context.new_page()
        page.set_default_timeout(30000)

        print("✅ Browser ready with download support")
        print()

        # Navigate to website
        print("🌐 Loading text-to-speech.online...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(3)

        # Handle cookie consent if present
        print("🍪 Checking for consent dialog...")
        try:
            # Try multiple consent button selectors
            consent_selectors = [
                'button:has-text("I agree")',
                'button:has-text("Accept")',
                'button:has-text("Consent")',
                '.fc-button.fc-cta-consent',  # Consent Manager specific
                'button[aria-label*="consent" i]'
            ]

            consent_handled = False
            for selector in consent_selectors:
                try:
                    consent_btn = page.locator(selector).first
                    if await consent_btn.is_visible(timeout=2000):
                        await consent_btn.click()
                        print(f"✅ Consent accepted using: {selector}")
                        await asyncio.sleep(3)
                        consent_handled = True
                        break
                except:
                    continue

            if not consent_handled:
                print("ℹ️  No consent dialog found or already handled")
        except Exception as e:
            print(f"ℹ️  Consent handling: {e}")

        print()

        # Select locale: English (South Africa)
        print("🌍 Selecting locale: English (South Africa)...")
        try:
            # Wait for first select (locale dropdown)
            locale_select = page.locator('select').first
            await locale_select.wait_for(state='visible', timeout=10000)

            # Select English (South Africa)
            await locale_select.select_option(label="English (South Africa) [en-ZA]")
            print("✅ Locale selected: English (South Africa)")
            print("⏳ Waiting for voice options to reload...")
            await asyncio.sleep(5)  # Wait longer for voice options to reload
        except Exception as e:
            print(f"⚠️  Locale selection failed: {e}")
            print("   Continuing with default locale...")

        # Select male voice
        print("🎤 Selecting male voice...")
        try:
            # Get all selects, second one is voice
            selects = await page.locator('select').all()

            if len(selects) >= 2:
                voice_select = selects[1]

                # Get all options and find male voice
                options = await voice_select.locator('option').all()
                male_voice_found = False

                print(f"   Available voices (total: {len(options)}):")
                for idx, option in enumerate(options):
                    text = await option.text_content()
                    # Show all voices for debugging
                    if idx < 10:  # Show first 10
                        print(f"   {idx+1}. {text}")

                    # Select first male voice found, but EXCLUDE female voices
                    if not male_voice_found:
                        text_lower = text.lower()
                        # Check if it's male AND not female
                        is_male = any(keyword in text_lower for keyword in ['male', 'guy', 'man', 'luke'])
                        is_female = 'female' in text_lower

                        if is_male and not is_female:
                            value = await option.get_attribute('value')
                            await voice_select.select_option(value=value)
                            print(f"✅ Selected male voice: {text}")
                            male_voice_found = True
                            await asyncio.sleep(2)
                            break

                if not male_voice_found:
                    print("⚠️  No male voice found, using default")
            else:
                print("⚠️  Voice dropdown not found")

        except Exception as e:
            print(f"⚠️  Voice selection failed: {e}")
            print("   Continuing with default voice...")

        print()

        # Enter text
        print("⌨️  Entering text...")
        try:
            textarea = page.locator('textarea').first
            await textarea.wait_for(state='visible', timeout=10000)
            await textarea.clear()
            await textarea.fill(test_text)
            print("✅ Text entered successfully")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Failed to enter text: {e}")
            return False

        # Generate speech
        print("🔊 Generating speech...")
        try:
            # Use the ID we found: #quick-play
            play_btn = page.locator('#quick-play')
            await play_btn.wait_for(state='visible', timeout=10000)
            await play_btn.click()
            print("✅ Clicked play button: #quick-play")

            # Wait for speech generation
            print("⏳ Waiting for speech generation (15 seconds)...")
            await asyncio.sleep(15)

        except Exception as e:
            print(f"❌ Failed to generate speech: {e}")
            return False

        print()

        # Download audio
        print("⬇️  Downloading audio...")
        try:
            # Use the ID we found: #download
            download_btn = page.locator('#download')
            await download_btn.wait_for(state='visible', timeout=10000)
            print("✅ Found download button: #download")

            # Wait for download button to be enabled (not disabled)
            print("⏳ Waiting for download button to become enabled...")
            await page.wait_for_function(
                "document.querySelector('#download') && !document.querySelector('#download').disabled",
                timeout=30000
            )
            print("✅ Download button is enabled")
            await asyncio.sleep(1)

            # Setup download capture BEFORE clicking
            print("🎯 Setting up download capture...")
            async with page.expect_download(timeout=60000) as download_info:
                await download_btn.click()
                print("✅ Download button clicked")

            download = await download_info.value
            suggested_filename = download.suggested_filename
            print(f"📦 Download started: {suggested_filename}")

            # Save to our directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_filename = f"tts_audio_{timestamp}.mp3"
            target_path = output_dir / target_filename

            print(f"💾 Saving to: {target_path}")
            await download.save_as(str(target_path))

            # Verify download
            await asyncio.sleep(2)

            if target_path.exists():
                file_size = target_path.stat().st_size
                print()
                print("=" * 70)
                print("🎉 DOWNLOAD SUCCESSFUL!")
                print("=" * 70)
                print(f"📄 File: {target_filename}")
                print(f"📍 Path: {target_path.absolute()}")
                print(f"📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                print(f"📁 Directory: {output_dir.absolute()}")
                print()

                if file_size > 1000:
                    print("✅ File size looks good!")
                    return True
                else:
                    print(f"⚠️  Warning: Small file size ({file_size} bytes)")
                    return False
            else:
                print("❌ File not found at expected location")
                return False

        except Exception as e:
            print(f"❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print()
        print("🧹 Cleaning up...")
        try:
            if 'context' in locals():
                await context.close()
            if 'browser' in locals():
                await browser.close()
            if 'playwright' in locals():
                await playwright.stop()
            print("✅ Cleanup complete")
        except:
            pass

async def main():
    """Main entry point"""
    print()
    print("🚀 TTS Online Audio Downloader")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    success = await download_tts_audio()

    print()
    print("=" * 70)
    if success:
        print("✅ SUCCESS - Audio file downloaded to debug_download2/")
    else:
        print("❌ FAILED - No audio file downloaded")
    print("=" * 70)
    print()

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
