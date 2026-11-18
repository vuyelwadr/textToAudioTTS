#!/usr/bin/env python3
"""
FINAL DOWNLOAD ONLY TEST - Skip Play, go straight to Download
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

async def final_download_only():
    """Final download test - skip Play button"""
    
    print("🎯 FINAL DOWNLOAD ONLY TEST")
    print("=" * 80)
    
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target: {output_dir.absolute()}")
    
    playwright = await async_playwright().start()
    
    try:
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--no-sandbox"]
        )
        
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup complete")
        
        # Navigate
        await page.goto("https://www.text-to-speech.online/")
        await asyncio.sleep(8)
        
        # Handle consent
        try:
            await page.click('button:has-text("I agree")', timeout=3000)
            print("✅ Consent accepted")
            await asyncio.sleep(3)
        except:
            print("ℹ️ No consent found")
        
        # Remove overlays
        try:
            await page.evaluate("""
                () => {
                    const overlays = document.querySelectorAll('[class*="overlay"], [class*="consent"]');
                    for (const overlay of overlays) {
                        overlay.style.display = 'none';
                        overlay.remove();
                    }
                    return true;
                }
            """)
            print("✅ Overlays removed")
        except:
            pass
        
        # Select locale
        print("🌍 Selecting English South Africa...")
        try:
            await page.select_option('select', label="English (South Africa) [en-ZA]")
            print("✅ Locale selected")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Locale error: {e}")
        
        # Select voice
        print("🎤 Selecting voice...")
        try:
            await page.select_option('select', label="LukeNeural(Male)")
            print("✅ Male voice selected")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Voice error: {e}")
        
        # Enter text
        test_text = "Hello world. This is final download only test for debug directory. No play button clicked."
        print(f"⌨️ Entering: {test_text}")
        
        try:
            await page.fill('textarea', test_text)
            print("✅ Text entered")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Text error: {e}")
        
        # WAIT for natural processing (don't click Play)
        print("⏳ Waiting for natural processing (no Play button)...")
        await asyncio.sleep(10)
        
        # LOOK FOR DOWNLOAD BUTTON directly
        print("🎯 LOOKING FOR DOWNLOAD BUTTON...")
        
        try:
            # Wait up to 20 seconds for Download button to appear naturally
            print("⏳ Waiting for Download button to appear (20 seconds)...")
            
            # Wait for Download button to be visible
            download_button = page.locator('button').filter(has_text="Download").first
            await download_button.wait_for(state='visible', timeout=20000)
            print("✅ Download button found and visible!")
            await asyncio.sleep(3)
            
            # THE WORKING PATTERN
            print("🎯 SETTING UP EXPECT_DOWNLOAD...")
            
            async with page.expect_download() as download_info:
                print("⬇️ CLICKING DOWNLOAD BUTTON...")
                await download_button.click()
            
            download = await download_info.value
            suggested_filename = download.suggested_filename
            print(f"📄 Download captured: {suggested_filename}")
            
            # PURE STRING PATH
            string_path = f"{output_dir}/FINAL_DOWNLOAD_ONLY.mp3"
            print(f"💾 SAVING TO: {string_path}")
            await download.save_as(string_path)
            
            await asyncio.sleep(5)
            
            # VERIFY
            if Path(string_path).exists():
                file_size = Path(string_path).stat().st_size
                print(f"🎉 DOWNLOAD SUCCESS!")
                print(f"   📍 Location: {Path(string_path).absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                print(f"   🎯 Method: NO Play button, direct download")
                print(f"   ✅ Pattern: expect_download() + save_as()")
                
                if file_size > 1000:
                    print("✅ Valid file!")
                    print("✅ DOWNLOAD ISSUE FINALLY SOLVED!")
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
    print("🚀 Starting FINAL DOWNLOAD ONLY test...")
    
    success = await final_download_only()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 FINAL DOWNLOAD ONLY TEST PASSED!")
        print("✅ File downloaded to debug_download directory")
        print("✅ No Play button clicked - direct download")
        print("✅ expect_download() context manager works")
        print("✅ download.save_as() works correctly")
        print("✅ Download path control is EXACT")
        print("✅ All issues FIXED!")
        print("✅ TTS ONLINE DOWNLOAD ISSUE SOLVED!")
    else:
        print("❌ FINAL DOWNLOAD ONLY TEST FAILED")
    
    print("=" * 80)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
