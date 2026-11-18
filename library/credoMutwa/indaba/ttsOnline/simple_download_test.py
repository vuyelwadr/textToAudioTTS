#!/usr/bin/env python3
"""
SIMPLE DOWNLOAD TEST - Robust version with longer waits
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

async def simple_download_test():
    """Simple robust download test"""
    
    print("🧪 SIMPLE DOWNLOAD TEST")
    print("=" * 50)
    
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target: {output_dir.absolute()}")
    
    playwright = await async_playwright().start()
    
    try:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        # Navigate
        print("🌐 Navigating...")
        await page.goto("https://www.text-to-speech.online/")
        await asyncio.sleep(5)
        
        # Accept consent
        try:
            await page.click('button:has-text("I agree")', timeout=3000)
            print("✅ Consent accepted")
            await asyncio.sleep(2)
        except:
            pass
        
        # Select locale
        print("🌍 Selecting English South Africa...")
        await page.select_option('select', "English (South Africa) [en-ZA]")
        await asyncio.sleep(3)
        
        # Select voice  
        print("🎤 Selecting male voice...")
        await page.select_option('select:nth-child(2)', "LukeNeural(Male)")
        await asyncio.sleep(3)
        
        # Enter text
        print("⌨️ Entering test text...")
        await page.fill('textarea', "Hello world. This is a test download to debug directory.")
        await asyncio.sleep(2)
        
        # Click Play and WAIT LONGER
        print("🔊 Clicking Play button...")
        await page.click('button:has-text("Play")')
        print("⏳ Waiting for speech generation (10 seconds)...")
        await asyncio.sleep(10)
        
        # Look for Download button
        print("🔍 Looking for Download button...")
        try:
            await page.wait_for_selector('button:has-text("Download")', timeout=15000)
            print("✅ Download button found!")
            await asyncio.sleep(2)
            
            # Setup download capture
            print("🎯 Setting up download capture...")
            download_promise = page.wait_for_event('download')
            
            # Click Download
            print("⬇️ Clicking Download...")
            await page.click('button:has-text("Download")')
            
            # Wait for download
            print("⏳ Waiting for download...")
            download = await asyncio.wait_for(download_promise, timeout=30.0)
            
            # Save to debug_download
            target_path = output_dir / "simple_test.mp3"
            print(f"💾 Saving to: {target_path}")
            await download.save_as(target_path)
            
            await asyncio.sleep(2)
            
            if target_path.exists():
                size = target_path.stat().st_size
                print(f"🎉 SUCCESS! File downloaded!")
                print(f"📍 Location: {target_path}")
                print(f"📊 Size: {size:,} bytes")
                return True
            else:
                print("❌ File not found")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
        
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()

async def main():
    print("🚀 Starting simple download test...")
    
    success = await simple_download_test()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 DOWNLOAD TEST PASSED!")
        print("✅ Check debug_download/simple_test.mp3")
    else:
        print("❌ DOWNLOAD TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
