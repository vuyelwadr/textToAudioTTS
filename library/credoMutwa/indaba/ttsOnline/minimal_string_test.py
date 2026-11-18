#!/usr/bin/env python3
"""
MINIMAL STRING TEST - Isolate string conversion issue
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

async def minimal_string_test():
    """Minimal test with explicit string conversion"""
    
    print("🔧 MINIMAL STRING TEST")
    print("=" * 60)
    
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target: {output_dir.absolute()}")
    
    playwright = await async_playwright().start()
    
    try:
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        # Navigate
        await page.goto("https://www.text-to-speech.online/")
        await asyncio.sleep(5)
        
        # Try direct download test with a simple site first
        print("🧪 Testing with simple PDF download...")
        await page.goto("https://www.buds.com.ua/images/Lorem_ipsum.pdf")
        await asyncio.sleep(3)
        
        # Simple PDF download test
        print("🎯 Testing PDF download...")
        download_promise = page.wait_for_event('download')
        
        # Click PDF link
        await page.click('a[href*="Lorem_ipsum.pdf"]')
        
        print("⏳ Waiting for PDF download...")
        download = await asyncio.wait_for(download_promise, timeout=30.0)
        
        # EXPLICIT STRING PATH
        string_path = f"{output_dir}/test.pdf"  # String literal
        print(f"💾 Saving to EXACT string path: {string_path}")
        
        # Test the save_as with string
        await download.save_as(string_path)
        
        await asyncio.sleep(2)
        
        if Path(string_path).exists():
            file_size = Path(string_path).stat().st_size
            print(f"🎉 PDF DOWNLOAD WORKS!")
            print(f"   📍 Path: {Path(string_path).absolute()}")
            print(f"   📊 Size: {file_size:,} bytes")
            print(f"   🔧 String path test: PASSED")
            return True
        else:
            print("❌ PDF download failed")
            return False
        
    finally:
        try:
            await context.close()
            await browser.close()
            await playwright.stop()
        except:
            pass

async def main():
    print("🚀 Starting minimal string test...")
    
    success = await minimal_string_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MINIMAL STRING TEST PASSED!")
        print("✅ download.save_as() works with string paths")
        print("✅ String conversion is correct")
    else:
        print("❌ MINIMAL STRING TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
