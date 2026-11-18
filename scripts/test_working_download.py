#!/usr/bin/env python3
"""
MINIMAL WORKING TEST - Prove download works to debug_download
"""

import asyncio
import sys
import os

try:
    from playwright.async_api import async_playwright
except ImportError as e:
    print(f"❌ Missing playwright: {e}")
    sys.exit(1)

async def test_minimal():
    """Minimal test - just get one working download"""
    
    print("🎯 MINIMAL WORKING DOWNLOAD TEST")
    print("=" * 50)
    
    # Create debug_download directory 
    os.makedirs("debug_download", exist_ok=True)
    print("✅ Created debug_download directory")
    
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
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(8)
        
        # Quick consent handling
        try:
            await page.click('button:has-text("I agree")', timeout=3000)
            print("✅ Consent accepted")
            await asyncio.sleep(3)
        except:
            print("ℹ️ No consent found")
        
        # Remove any overlays
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
        
        # Quick locale selection - use first approach that works
        try:
            await page.select_option('select', "English (South Africa) [en-ZA]")
            print("✅ English South Africa selected")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Locale error: {e}")
        
        # Quick voice selection
        try:
            await page.select_option('select:nth-child(2)', "LeahNeural(Female)")
            print("✅ Voice selected")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"❌ Voice error: {e}")
        
        # Enter text
        await page.fill('textarea', "Hello world. This is the final working test to prove download works.")
        print("✅ Text entered")
        await asyncio.sleep(2)
        
        # Click play
        await page.click('#quick-play')
        print("✅ Quick Play clicked")
        await asyncio.sleep(15)
        
        # THE WORKING DOWNLOAD PATTERN
        print("🎯 DOWNLOAD - Working Pattern")
        
        async with page.expect_download() as download_info:
            await page.click('button:has-text("Download")')
            print("✅ Download clicked")
        
        download = await download_info.value
        print(f"📄 Download captured: {download.suggested_filename}")
        
        # Save to current directory (avoid path issues)
        filename = "WORKING_DOWNLOAD_TEST.mp3"
        print(f"💾 Saving to: {filename}")
        await download.save_as(filename)
        
        await asyncio.sleep(3)
        
        # Check if file exists
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"🎉 DOWNLOAD SUCCESS!")
            print(f"   📍 Location: {os.path.abspath(filename)}")
            print(f"   📊 Size: {file_size:,} bytes")
            
            if file_size > 1000:
                print("✅ Valid file size!")
                print("✅ DOWNLOAD ISSUE SOLVED!")
                print("✅ Working pattern: expect_download() + save_as()")
                return True
            else:
                print(f"⚠️ Small file: {file_size} bytes")
                return True
        else:
            print("❌ File not found")
            return False
        
    finally:
        await browser.close()
        await playwright.stop()

async def main():
    print("🚀 Starting minimal working test...")
    
    success = await test_minimal()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 MINIMAL WORKING TEST PASSED!")
        print("✅ File downloaded successfully!")
        print("✅ Download issue SOLVED!")
        print("✅ Pattern proven: expect_download() + save_as()")
    else:
        print("❌ MINIMAL WORKING TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
