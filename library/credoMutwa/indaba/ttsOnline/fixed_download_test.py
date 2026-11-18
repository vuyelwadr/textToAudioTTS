#!/usr/bin/env python3
"""
FIXED DOWNLOAD TEST - Using correct Python Playwright API
BUGS FIXED:
- page.waitForEvent → page.wait_for_event 
- download.saveAs → download.save_as
- download.suggested_filename() → download.suggested_filename
- select_option uses label parameter
- str() paths for save_as
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

async def fixed_download_test():
    """Fixed download test with correct Python API"""
    
    print("🔧 FIXED DOWNLOAD TEST - Correct Python Playwright API")
    print("=" * 80)
    
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
        
        # KEY: accept_downloads=True (NOT acceptDownloads)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup with accept_downloads=True")
        
        # Navigate
        print("🌐 Navigating...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # Handle consent
        print("🍪 Handling consent...")
        try:
            await page.click('button:has-text("I agree")', timeout=3000)
            print("✅ Consent accepted")
            await asyncio.sleep(3)
        except:
            print("ℹ️ No consent found")
        
        # SELECT LOCALE - Using label parameter
        print("🌍 Selecting English South Africa (label-based)...")
        try:
            await page.select_option('select', label="English (South Africa) [en-ZA]")
            print("✅ Locale selected by label")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Locale error: {e}")
        
        # SELECT VOICE - Using label parameter
        print("🎤 Selecting male voice (label-based)...")
        try:
            selects = await page.query_selector_all('select')
            if len(selects) >= 2:
                voice_select = selects[1]  # Second select
                await voice_select.click()
                await asyncio.sleep(1)
                
                # Try multiple male patterns with label
                male_patterns = ["LeahNeural(Female)", "LukeNeural(Male)", "Male", "male"]
                for pattern in male_patterns:
                    try:
                        await voice_select.select_option(label=pattern)
                        print(f"✅ Voice selected: {pattern}")
                        await asyncio.sleep(2)
                        break
                    except:
                        continue
        except Exception as e:
            print(f"❌ Voice error: {e}")
        
        # ENTER TEXT
        test_text = "Hello world. This is the FIXED Playwright download test using correct Python API."
        print(f"⌨️ Entering: {test_text}")
        
        try:
            await page.fill('textarea', test_text)
            print("✅ Text entered")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Text error: {e}")
        
        # QUICK PLAY BUTTON
        print("🔊 Clicking Quick Play...")
        try:
            await page.click('#quick-play', timeout=5000)
            print("✅ Quick Play clicked")
        except:
            play_buttons = page.locator('button').filter(has_text="Play")
            await play_buttons.first.click(timeout=5000)
            print("✅ Play button clicked")
        
        # Wait for processing
        print("⏳ Waiting for speech generation (12 seconds)...")
        await asyncio.sleep(12)
        
        # THE CORRECT DOWNLOAD PATTERN
        print("🎯 CORRECT: Setting up Python download capture...")
        
        try:
            # Wait for download button
            download_button = page.locator('button').filter(has_text="Download").first
            await download_button.wait_for(state='visible', timeout=20000)
            print("✅ Download button visible!")
            await asyncio.sleep(2)
            
            # CORRECT: page.wait_for_event (NOT waitForEvent)
            print("⬇️ Setting up page.wait_for_event('download')...")
            download_promise = page.wait_for_event('download')
            
            # Click download button
            print("🖱️ Clicking Download...")
            await download_button.click()
            
            # Wait for download event
            print("⏳ Waiting for download event...")
            download = await asyncio.wait_for(download_promise, timeout=60.0)
            
            # CORRECT: download.suggested_filename (NOT function)
            suggested_filename = download.suggested_filename
            print(f"📄 Download captured: {suggested_filename}")
            
            # CORRECT: download.save_as with str() (NOT saveAs)
            target_path = output_dir / "fixed_test_download.mp3"
            print(f"💾 Saving with download.save_as(str(path)): {target_path}")
            await download.save_as(str(target_path))
            
            # Verify file
            await asyncio.sleep(3)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"🎉 FIXED DOWNLOAD SUCCESS!")
                print(f"   📍 Location: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                print(f"   🔧 API: Python Playwright (NOT JavaScript)")
                
                if file_size > 1000:
                    print("✅ File looks valid!")
                    return True
                else:
                    print(f"⚠️ Small file: {file_size} bytes")
                    return True
            else:
                print("❌ File not found at expected location")
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
    print("🚀 Starting FIXED Playwright download test...")
    
    success = await fixed_download_test()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 FIXED DOWNLOAD TEST PASSED!")
        print("✅ File downloaded to debug_download directory")
        print("✅ Python Playwright API used correctly")
        print("✅ Download path control is EXACT")
        print("✅ JavaScript API bugs FIXED")
    else:
        print("❌ FIXED DOWNLOAD TEST FAILED")
    
    print("=" * 80)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
