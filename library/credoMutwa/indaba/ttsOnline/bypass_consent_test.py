#!/usr/bin/env python3
"""
BYPASS CONSENT DOWNLOAD TEST - Uses JavaScript to bypass overlay
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

async def bypass_consent_test():
    """Bypass consent overlay using JavaScript"""
    
    print("🛡️ BYPASS CONSENT DOWNLOAD TEST")
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
        
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("✅ Playwright setup complete")
        
        # Navigate
        print("🌐 Navigating...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(5)
        
        # BYPASS CONSENT - JavaScript approach
        print("🍪 Bypassing consent with JavaScript...")
        
        # Method 1: Click any consent button with JavaScript
        consent_js = """
        () => {
            // Find and click any consent button
            const buttons = document.querySelectorAll('button');
            const consentTexts = ['I agree', 'Consent', 'Accept', 'Accept all'];
            for (const button of buttons) {
                if (consentTexts.some(text => button.textContent.includes(text))) {
                    button.click();
                    return true;
                }
            }
            return false;
        }
        """
        
        try:
            consent_result = await page.evaluate(consent_js)
            if consent_result:
                print("✅ Consent clicked via JavaScript")
                await asyncio.sleep(3)
            else:
                print("ℹ️ No consent dialog found")
        except Exception as e:
            print(f"ℹ️ Consent bypass attempt: {e}")
        
        # Method 2: Force remove overlay if still there
        remove_overlay_js = """
        () => {
            // Remove any consent overlay
            const overlays = document.querySelectorAll('[class*="overlay"], [class*="consent"], [class*="modal"]');
            for (const overlay of overlays) {
                overlay.style.display = 'none';
                overlay.remove();
            }
            // Remove backdrop
            const backdrops = document.querySelectorAll('[class*="backdrop"]');
            for (const backdrop of backdrops) {
                backdrop.remove();
            }
            return true;
        }
        """
        
        try:
            await page.evaluate(remove_overlay_js)
            print("✅ Overlays removed via JavaScript")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"ℹ️ Overlay removal: {e}")
        
        # SELECT LOCALE - JavaScript injection
        print("🌍 Selecting locale with JavaScript...")
        locale_js = """
        () => {
            // Find locale select and select English South Africa
            const selects = document.querySelectorAll('select');
            for (const select of selects) {
                const options = select.options;
                for (let i = 0; i < options.length; i++) {
                    if (options[i].text.includes('English (South Africa)')) {
                        select.selectedIndex = i;
                        select.dispatchEvent(new Event('change'));
                        return true;
                    }
                }
            }
            return false;
        }
        """
        
        try:
            locale_result = await page.evaluate(locale_js)
            if locale_result:
                print("✅ Locale selected via JavaScript")
                await asyncio.sleep(3)
            else:
                print("❌ Locale selection failed")
        except Exception as e:
            print(f"❌ Locale JS error: {e}")
        
        # SELECT VOICE - JavaScript injection
        print("🎤 Selecting voice with JavaScript...")
        voice_js = """
        () => {
            // Find voice select (second one) and select male
            const selects = document.querySelectorAll('select');
            if (selects.length >= 2) {
                const voiceSelect = selects[1];
                const options = voiceSelect.options;
                for (let i = 0; i < options.length; i++) {
                    if (options[i].text.toLowerCase().includes('male')) {
                        voiceSelect.selectedIndex = i;
                        voiceSelect.dispatchEvent(new Event('change'));
                        return options[i].text;
                    }
                }
                // Fallback to first non-female
                for (let i = 0; i < options.length; i++) {
                    if (!options[i].text.toLowerCase().includes('female')) {
                        voiceSelect.selectedIndex = i;
                        voiceSelect.dispatchEvent(new Event('change'));
                        return options[i].text;
                    }
                }
            }
            return null;
        }
        """
        
        try:
            voice_result = await page.evaluate(voice_js)
            if voice_result:
                print(f"✅ Voice selected via JavaScript: {voice_result}")
                await asyncio.sleep(3)
            else:
                print("❌ Voice selection failed")
        except Exception as e:
            print(f"❌ Voice JS error: {e}")
        
        # ENTER TEXT - JavaScript
        print("⌨️ Entering text with JavaScript...")
        text_js = """
        (text) => {
            const textarea = document.querySelector('textarea');
            if (textarea) {
                textarea.value = text;
                textarea.dispatchEvent(new Event('input'));
                return true;
            }
            return false;
        }
        """
        
        test_text = "Hello world. This is JavaScript bypass consent overlay download test."
        try:
            text_result = await page.evaluate(text_js, test_text)
            if text_result:
                print("✅ Text entered via JavaScript")
                await asyncio.sleep(2)
            else:
                print("❌ Text entry failed")
        except Exception as e:
            print(f"❌ Text JS error: {e}")
        
        # QUICK PLAY - JavaScript
        print("🔊 Clicking Quick Play with JavaScript...")
        play_js = """
        () => {
            // Find quick play button
            const buttons = document.querySelectorAll('button');
            for (const button of buttons) {
                if (button.id === 'quick-play' || button.textContent.includes('Play')) {
                    button.click();
                    return true;
                }
            }
            return false;
        }
        """
        
        try:
            play_result = await page.evaluate(play_js)
            if play_result:
                print("✅ Quick Play clicked via JavaScript")
                await asyncio.sleep(15)  # Wait for speech generation
            else:
                print("❌ Quick Play failed")
        except Exception as e:
            print(f"❌ Play JS error: {e}")
        
        # DOWNLOAD - JavaScript + Correct Python API
        print("🎯 JavaScript + Correct Python Download API...")
        
        try:
            # Check if download button exists
            download_check_js = """
            () => {
                const buttons = document.querySelectorAll('button');
                for (const button of buttons) {
                    if (button.textContent.includes('Download')) {
                        return button.textContent;
                    }
                }
                return null;
            }
            """
            
            download_button_text = await page.evaluate(download_check_js)
            if download_button_text:
                print(f"✅ Download button found: {download_button_text}")
            else:
                print("❌ No download button found")
                return False
            
            # THE CORRECT PATTERN
            print("🎯 Setting up Python download capture...")
            download_promise = page.wait_for_event('download')
            
            # Click download with JavaScript
            click_download_js = """
            () => {
                const buttons = document.querySelectorAll('button');
                for (const button of buttons) {
                    if (button.textContent.includes('Download')) {
                        button.click();
                        return true;
                    }
                }
                return false;
            }
            """
            
            click_result = await page.evaluate(click_download_js)
            if click_result:
                print("✅ Download clicked via JavaScript")
            else:
                print("❌ Download click failed")
            
            # Wait for download event
            print("⏳ Waiting for download event...")
            download = await asyncio.wait_for(download_promise, timeout=60.0)
            
            # Save correctly
            suggested_filename = download.suggested_filename
            target_path = output_dir / "bypass_test_download.mp3"
            print(f"💾 Saving: {target_path}")
            await download.save_as(str(target_path))
            
            await asyncio.sleep(3)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"🎉 BYPASS DOWNLOAD SUCCESS!")
                print(f"   📍 Location: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                print(f"   📁 Directory: debug_download/")
                
                if file_size > 1000:
                    print("✅ Valid file size!")
                    print("✅ Consent overlay bypassed!")
                    print("✅ JavaScript + Python API combo WORKS!")
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
    print("🚀 Starting BYPASS CONSENT download test...")
    
    success = await bypass_consent_test()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 BYPASS CONSENT DOWNLOAD TEST PASSED!")
        print("✅ File downloaded to debug_download directory")
        print("✅ Consent overlay bypassed with JavaScript")
        print("✅ Python Playwright API works correctly")
        print("✅ Download path control is EXACT")
    else:
        print("❌ BYPASS CONSENT DOWNLOAD TEST FAILED")
    
    print("=" * 80)
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
