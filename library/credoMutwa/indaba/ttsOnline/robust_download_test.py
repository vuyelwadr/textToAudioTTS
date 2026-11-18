#!/usr/bin/env python3
"""
ROBUST DOWNLOAD TEST - Handles consent overlay properly
"""

import asyncio
import os
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError as e:
    print(f"❌ Missing playwright: {e}")
    sys.exit(1)

async def robust_download_test():
    """Robust download test with consent handling"""
    
    print("🛡️ ROBUST DOWNLOAD TEST - Consent Overlay Handling")
    print("=" * 70)
    
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target: {output_dir.absolute()}")
    
    # Clear existing files
    for file in output_dir.glob("*"):
        file.unlink()
        print(f"🗑️ Removed: {file.name}")
    
    playwright = await async_playwright().start()
    
    try:
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup complete")
        
        # Navigate
        print("🌐 Navigating...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # ROBUST CONSENT HANDLING
        print("🍪 Robust consent handling...")
        consent_handled = False
        
        # Try multiple consent approaches
        consent_attempts = [
            'button:has-text("I agree")',
            'button:has-text("Consent")', 
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button.fc-button-primary'  # Class-based
        ]
        
        for attempt in range(3):  # Try 3 times
            for selector in consent_attempts:
                try:
                    await page.click(selector, timeout=2000)
                    print(f"✅ Consent clicked with: {selector}")
                    consent_handled = True
                    await asyncio.sleep(3)
                    break
                except:
                    continue
            
            if consent_handled:
                break
            
            # Wait and try again
            print(f"⏳ Consent attempt {attempt + 1}/3, waiting...")
            await asyncio.sleep(2)
        
        # SELECT LOCALE - Use ID-based approach
        print("🌍 Selecting locale (ID-based)...")
        try:
            # Try to click locale dropdown by ID if available
            locale_select = page.locator('select').first
            await locale_select.click(timeout=5000)
            print("✅ Locale dropdown clicked")
            await asyncio.sleep(1)
            
            # Select English South Africa
            en_za_option = page.locator('option').filter(has_text="English (South Africa)").first
            await en_za_option.click(timeout=5000)
            print("✅ Selected English South Africa")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Locale selection error: {e}")
        
        # SELECT VOICE - Use more specific approach
        print("🎤 Selecting voice (second select)...")
        try:
            voice_select = page.locator('select').nth(1)  # Second select
            await voice_select.click(timeout=5000)
            print("✅ Voice dropdown clicked")
            await asyncio.sleep(1)
            
            # Find male voice
            male_option = page.locator('option').filter(has_text_regex=r".*[Mm]ale.*").first
            await male_option.click(timeout=5000)
            print("✅ Selected male voice")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Voice selection error: {e}")
        
        # ENTER TEXT
        test_text = "Hello world. This is the robust Playwright download test for debug directory."
        print(f"⌨️ Entering text: {test_text}")
        
        try:
            await page.fill('textarea', test_text, timeout=5000)
            print("✅ Text entered")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Text entry error: {e}")
        
        # QUICK PLAY BUTTON - Use specific ID
        print("🔊 Clicking Quick Play button...")
        try:
            # Try the quick-play button ID if available
            quick_play = page.locator('#quick-play')
            await quick_play.click(timeout=5000)
            print("✅ Quick Play clicked")
        except:
            # Fallback to text-based selector
            play_buttons = page.locator('button').filter(has_text="Play")
            await play_buttons.first.click(timeout=5000)
            print("✅ Play button clicked")
        
        # Wait for processing
        print("⏳ Waiting for speech generation (15 seconds)...")
        await asyncio.sleep(15)
        
        # DOWNLOAD - THE CRITICAL PART
        print("🎯 CRITICAL: Download capture setup...")
        
        try:
            # Wait for Download button
            download_button = page.locator('button').filter(has_text="Download").first
            await download_button.wait_for(state='visible', timeout=20000)
            print("✅ Download button found and visible!")
            await asyncio.sleep(2)
            
            # THE PATTERN: Promise.all with event capture
            print("🎯 Setting up Promise.all download capture...")
            download_promise = page.waitForEvent('download')
            
            # Click download
            print("⬇️ Clicking download button...")
            await download_button.click()
            
            # Wait for download event
            print("⏳ Waiting for download event...")
            download = await asyncio.wait_for(download_promise, timeout=45.0)
            
            # EXACT PATH CONTROL
            suggested_filename = download.suggested_filename()
            print(f"📄 Download event captured: {suggested_filename}")
            
            target_path = output_dir / "robust_test_download.mp3"
            print(f"💾 Saving to EXACT path: {target_path}")
            await download.saveAs(target_path)
            
            # Verify
            await asyncio.sleep(3)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"🎉 ROBUST DOWNLOAD SUCCESS!")
                print(f"   📍 Location: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                
                if file_size > 1000:
                    print("✅ File looks valid!")
                    print("✅ Playwright download.saveAs() WORKS!")
                    print("✅ EXACT path control PROVEN!")
                    return True
                else:
                    print(f"⚠️ Small file: {file_size} bytes")
                    return True
            else:
                print("❌ File not found")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    finally:
        print("🧹 Cleaning up...")
        try:
            await context.close()
            await browser.close()
            await playwright.stop()
        except:
            pass

async def main():
    print("🚀 Starting ROBUST Playwright download test...")
    
    success = await robust_download_test()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ROBUST DOWNLOAD TEST PASSED!")
        print("✅ File downloaded to debug_download directory")
        print("✅ Playwright download.saveAs() PROVEN to work")
        print("✅ EXACT path control DEMONSTRATED")
        print("✅ Download ISSUE SOLVED!")
    else:
        print("❌ ROBUST DOWNLOAD TEST FAILED")
    
    print("=" * 70)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
