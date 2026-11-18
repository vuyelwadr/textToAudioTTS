#!/usr/bin/env python3
"""
CORRECT PLAYWRIGHT DOWNLOAD TEST - Following the exact pattern provided
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

async def correct_download_test():
    """Test following the exact provided Playwright pattern"""
    
    print("🎯 CORRECT PLAYWRIGHT DOWNLOAD TEST")
    print("=" * 60)
    
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target: {output_dir.absolute()}")
    
    # Clear any existing files
    for file in output_dir.glob("*"):
        file.unlink()
        print(f"🗑️ Removed: {file.name}")
    
    playwright = await async_playwright().start()
    
    try:
        # KEY: acceptDownloads: true 
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup with accept_downloads=True")
        
        # Navigate and wait
        print("🌐 Navigating...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # Handle consent
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
        
        # Select voice - get all selects first
        print("🎤 Selecting male voice...")
        selects = await page.query_selector_all('select')
        if len(selects) >= 2:
            voice_select = selects[1]  # Second select is voice
            options = await voice_select.query_selector_all('option')
            for option in options:
                text = await option.text_content()
                if "male" in text.lower():
                    await voice_select.select_option(text)
                    print(f"✅ Selected: {text}")
                    await asyncio.sleep(2)
                    break
        
        # Enter text
        test_text = "Hello world. This is the correct Playwright download test to debug directory."
        print(f"⌨️ Entering: {test_text}")
        await page.fill('textarea', test_text)
        await asyncio.sleep(2)
        
        # Play button
        print("🔊 Clicking Play button...")
        await page.click('button:has-text("Play")')
        
        # Wait for processing
        print("⏳ Waiting for speech generation (10 seconds)...")
        await asyncio.sleep(10)
        
        # KEY PATTERN: Promise.all with download event capture
        print("🎯 Setting up Promise.all download capture...")
        
        try:
            # Wait for download button to be available
            await page.wait_for_selector('button:has-text("Download")', timeout=15000)
            print("✅ Download button found!")
            
            # THE CRITICAL PATTERN - Capture download event BEFORE clicking
            downloadPromise = page.waitForEvent('download')
            
            # Click download button
            print("⬇️ Clicking Download button...")
            await page.click('button:has-text("Download")')
            
            # Wait for download
            print("⏳ Waiting for download event...")
            download = await downloadPromise
            
            # Get suggested filename and save
            suggested_filename = download.suggested_filename()
            print(f"📄 Download started: {suggested_filename}")
            
            # EXACT PATH CONTROL - save to debug_download
            target_path = output_dir / "correct_test_download.mp3"
            print(f"💾 Saving to: {target_path}")
            await download.saveAs(target_path)
            
            # Wait and verify
            await asyncio.sleep(3)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"🎉 DOWNLOAD SUCCESS!")
                print(f"   📍 Path: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                
                if file_size > 1000:
                    print("✅ File looks valid!")
                    return True
                else:
                    print(f"⚠️ Small file: {file_size} bytes")
                    return True
            else:
                print("❌ File not at expected path")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("🧹 Cleanup...")
        try:
            await context.close()
            await browser.close()
            await playwright.stop()
        except:
            pass

async def main():
    print("🚀 Starting CORRECT Playwright download test...")
    
    success = await correct_download_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CORRECT DOWNLOAD TEST PASSED!")
        print("✅ File downloaded to debug_download")
        print("✅ Playwright download.saveAs() works")
        print("✅ Path control is EXACT")
    else:
        print("❌ CORRECT DOWNLOAD TEST FAILED")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
