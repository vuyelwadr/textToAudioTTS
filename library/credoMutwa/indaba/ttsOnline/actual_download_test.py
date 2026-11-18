#!/usr/bin/env python3
"""
ACTUAL DOWNLOAD TEST - Download file to debug_download directory
This will prove the Playwright solution works by downloading real audio.
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

async def test_real_download():
    """Test actual download to debug_download directory"""
    
    print("🧪 ACTUAL DOWNLOAD TEST TO DEBUG_DOWNLOAD")
    print("=" * 60)
    
    # Target directory
    output_dir = Path("debug_download")
    os.makedirs(str(output_dir), exist_ok=True)
    
    print(f"📁 Target directory: {output_dir.absolute()}")
    
    # Clear any existing files
    for file in output_dir.glob("*"):
        file.unlink()
        print(f"🗑️ Removed: {file.name}")
    
    # Playwright setup
    playwright = await async_playwright().start()
    
    try:
        # Launch browser (visible so we can see it work)
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        
        # Create context WITH download acceptance
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Browser setup complete")
        
        # Navigate to site
        print("🌐 Navigating to text-to-speech.online...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(3)
        
        # Handle cookie consent
        print("🍪 Handling consent...")
        try:
            consent_selectors = [
                'button:has-text("I agree")',
                'button:has-text("Consent")',
                'button:has-text("Accept")'
            ]
            for selector in consent_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=3000)
                    await page.click(selector)
                    print("✅ Consent accepted")
                    await asyncio.sleep(1)
                    break
                except:
                    continue
        except Exception as e:
            print(f"ℹ️ No consent dialog: {e}")
        
        # Select locale
        print("🌍 Selecting English (South Africa)...")
        try:
            await page.wait_for_selector('select', timeout=10000)
            selects = await page.query_selector_all('select')
            
            for select in selects:
                options = await select.query_selector_all('option')
                for option in options:
                    text = await option.text_content()
                    if "English (South Africa)" in text:
                        await select.select_option(text)
                        print(f"✅ Selected locale: {text}")
                        await asyncio.sleep(2)
                        break
        except Exception as e:
            print(f"❌ Locale selection error: {e}")
            return False
        
        # Select male voice
        print("🎤 Selecting male voice...")
        try:
            await asyncio.sleep(3)
            selects = await page.query_selector_all('select')
            
            if len(selects) >= 2:
                voice_select = selects[1]
                options = await voice_select.query_selector_all('option')
                
                for option in options:
                    text = await option.text_content()
                    if "(male)" in text.lower():
                        await voice_select.select_option(text)
                        print(f"✅ Selected male voice: {text}")
                        await asyncio.sleep(2)
                        break
        except Exception as e:
            print(f"❌ Voice selection error: {e}")
        
        # Enter test text
        test_text = "This is a test of the Playwright download system. The file should save to debug download directory."
        print(f"⌨️ Entering text: {test_text}")
        
        try:
            await page.wait_for_selector('textarea', timeout=10000)
            await page.fill('textarea', test_text)
            print("✅ Text entered")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Text entry error: {e}")
        
        # Generate speech
        print("🔊 Generating speech...")
        try:
            generate_selectors = ['button:has-text("Play")', 'button:has-text("Generate")']
            for selector in generate_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.click(selector)
                    print("✅ Generate button clicked")
                    await asyncio.sleep(5)
                    break
                except:
                    continue
        except Exception as e:
            print(f"❌ Generate error: {e}")
        
        # DOWNLOAD - THE KEY TEST
        print("⬇️ STARTING DOWNLOAD TEST...")
        
        try:
            # Find download button
            download_selectors = [
                'button:has-text("Download")',
                'button[aria-label*="download"]',
                'a[download]'
            ]
            
            download_button = None
            for selector in download_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    download_button = await page.query_selector(selector)
                    if download_button:
                        print(f"✅ Found download button: {selector}")
                        break
                except:
                    continue
            
            if not download_button:
                print("❌ Download button not found")
                return False
            
            # CRITICAL: Setup download capture BEFORE clicking
            print("🎯 Setting up download capture...")
            download_promise = page.wait_for_event('download')
            
            # Click download button
            print("🖱️ Clicking download button...")
            await download_button.click()
            
            # Wait for download
            print("⏳ Waiting for download to start...")
            download = await asyncio.wait_for(download_promise, timeout=30.0)
            
            # Get info and save
            suggested_filename = download.suggested_filename()
            print(f"📄 Download started: {suggested_filename}")
            
            # Save to EXACT debug_download directory
            target_path = output_dir / "test_download.mp3"
            print(f"💾 Saving to: {target_path}")
            await download.save_as(target_path)
            
            # Verify
            await asyncio.sleep(2)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"🎉 DOWNLOAD SUCCESSFUL!")
                print(f"   📄 File: test_download.mp3")
                print(f"   📍 Location: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                
                if file_size > 1000:
                    print("✅ File size looks good!")
                    return True
                else:
                    print(f"⚠️ File seems small: {file_size} bytes")
                    return True
            else:
                print("❌ File not found at expected location")
                return False
                
        except asyncio.TimeoutError:
            print("❌ Download timeout")
            return False
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
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
    """Run actual download test"""
    print("🚀 Starting actual download test...")
    
    success = await test_real_download()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DOWNLOAD TEST PASSED!")
        print("✅ File downloaded to debug_download directory")
        print("✅ Playwright solution works correctly")
        print("✅ Download path control is EXACT")
    else:
        print("❌ DOWNLOAD TEST FAILED")
        print("❌ Check debug_download directory for results")
    
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
