#!/usr/bin/env python3
from pathlib import Path
"""
CURRENT DIRECTORY TEST - Download to current working directory
"""

import asyncio
import os
import sys

try:
    from playwright.async_api import async_playwright
except ImportError as e:
    print(f"❌ Missing playwright: {e}")
    sys.exit(1)

async def current_dir_test():
    """Test downloading to current working directory"""
    
    print("📁 CURRENT DIRECTORY DOWNLOAD TEST")
    print("=" * 60)
    
    current_dir = Path.cwd()
    print(f"📍 Working directory: {current_dir}")
    
    playwright = await async_playwright().start()
    
    try:
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--no-sandbox"]
        )
        
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup complete")
        
        # Test with simple PDF download first
        print("📄 Testing PDF download to current directory...")
        
        # Try a known working download
        await page.goto("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/pdf.pdf")
        await asyncio.sleep(3)
        
        # Download PDF to current directory
        print("🎯 Setting up PDF download...")
        
        async with page.expect_download() as download_info:
            await page.click('a[href*="pdf"]')  # PDF link
        
        download = await download_info.value
        suggested_filename = download.suggested_filename
        print(f"📄 PDF Download: {suggested_filename}")
        
        # Save to current directory (no subdirectories)
        current_path = f"{current_dir}/test_download.pdf"
        print(f"💾 Saving to: {current_path}")
        await download.save_as(current_path)
        
        await asyncio.sleep(3)
        
        if Path(current_path).exists():
            file_size = Path(current_path).stat().st_size
            print(f"🎉 PDF DOWNLOAD SUCCESS!")
            print(f"   📍 Location: {Path(current_path).absolute()}")
            print(f"   📊 Size: {file_size:,} bytes")
            print("✅ Download to CURRENT directory works!")
            return True
        else:
            print("❌ PDF file not found")
            return False
        
    finally:
        try:
            await context.close()
            await browser.close()
            await playwright.stop()
        except:
            pass

async def main():
    print("🚀 Starting current directory download test...")
    
    success = await current_dir_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CURRENT DIRECTORY DOWNLOAD TEST PASSED!")
        print("✅ download.save_as() works to current directory")
        print("✅ Playwright download API functional")
    else:
        print("❌ CURRENT DIRECTORY DOWNLOAD TEST FAILED")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
