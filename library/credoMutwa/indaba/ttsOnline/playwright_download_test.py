#!/usr/bin/env python3
"""
Playwright Download Test for TTS Online
Testing reliable download handling with explicit path control
"""

import asyncio
import time
import os
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install playwright")
    exit(1)

class PlaywrightTTSDownloader:
    def __init__(self, download_dir: str = "./tts_output"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.browser = None
        self.context = None
        self.page = None
        
    async def setup_browser(self):
        """Setup Playwright browser with download handling"""
        self.playwright = await async_playwright().start()
        
        # Launch browser
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Show browser for debugging
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-extensions",
                "--disable-dev-shm-usage",
                "--disable-gpu"
            ]
        )
        
        # Create context WITH download acceptance enabled
        # This is KEY difference from Selenium - explicit download control
        self.context = await self.browser.new_context(
            accept_downloads=True,  # Enable download handling
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        # Set timeout
        self.page.set_default_timeout(30000)
        
        print(f"✅ Playwright browser setup complete")
        print(f"📁 Configured download directory: {self.download_dir.absolute()}")
        return True
        
    async def handle_consent(self):
        """Handle cookie consent dialog"""
        try:
            print("🍪 Looking for cookie consent...")
            
            # Wait for consent dialog
            consent_selectors = [
                'button:has-text("I agree")',
                'button:has-text("Consent")',
                'button:has-text("Accept")',
                '.btn-primary:has-text("I agree")'
            ]
            
            for selector in consent_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    await self.page.click(selector)
                    print("✅ Cookie consent accepted")
                    await asyncio.sleep(2)
                    return True
                except:
                    continue
                    
            print("ℹ️ No consent dialog found")
            return True
            
        except Exception as e:
            print(f"⚠️ Consent handling error: {e}")
            return True
            
    async def select_locale(self, locale: str = "English (South Africa)"):
        """Select locale dropdown"""
        try:
            print(f"🌍 Selecting locale: {locale}")
            
            # Wait for locale dropdown
            await self.page.wait_for_selector('select', timeout=10000)
            
            # Get all selects and find locale dropdown
            selects = await self.page.query_selector_all('select')
            
            for select in selects:
                options = await select.query_selector_all('option')
                for option in options:
                    text = await option.text_content()
                    if locale in text:
                        await select.select_option(text)
                        print(f"✅ Selected locale: {text}")
                        await asyncio.sleep(2)
                        return True
            
            print(f"❌ Locale '{locale}' not found")
            return False
            
        except Exception as e:
            print(f"❌ Error selecting locale: {e}")
            return False
            
    async def select_voice(self, preferred_gender: str = "male"):
        """Select voice by gender preference"""
        try:
            print(f"🎤 Selecting {preferred_gender} voice")
            
            # Wait for page to update after locale selection
            await asyncio.sleep(3)
            
            # Get all selects and find voice dropdown (usually second one)
            selects = await self.page.query_selector_all('select')
            
            if len(selects) >= 2:
                voice_select = selects[1]
                options = await voice_select.query_selector_all('option')
                
                print("Available voices:")
                male_voice_found = False
                
                for option in options:
                    text = await option.text_content()
                    print(f"  - '{text}'")
                    
                    # Look for male voice explicitly
                    if not male_voice_found and "(male)" in text.lower():
                        await voice_select.select_option(text)
                        print(f"✅ Selected male voice: {text}")
                        male_voice_found = True
                        await asyncio.sleep(2)
                        return True
                
                # If no explicit male voice, try patterns
                if not male_voice_found:
                    male_patterns = ['male', 'guy', 'man', 'adam', 'mark', 'john']
                    for option in options:
                        text = await option.text_content()
                        if any(pattern in text.lower() for pattern in male_patterns):
                            await voice_select.select_option(text)
                            print(f"⚠ Selected male voice (pattern): {text}")
                            await asyncio.sleep(2)
                            return True
            
            print(f"❌ Could not find voice dropdown")
            return False
            
        except Exception as e:
            print(f"❌ Error selecting voice: {e}")
            return False
            
    async def enter_text(self, text: str):
        """Enter text into textarea"""
        try:
            print(f"⌨️ Entering text ({len(text)} characters)")
            
            # Wait for textarea
            await self.page.wait_for_selector('textarea', timeout=10000)
            
            # Clear and enter text
            await self.page.fill('textarea', text)
            print("✅ Text entered successfully")
            await asyncio.sleep(2)
            return True
            
        except Exception as e:
            print(f"❌ Error entering text: {e}")
            return False
            
    async def generate_speech(self):
        """Click generate button"""
        try:
            print("🔊 Generating speech...")
            
            # Look for generate/play button
            generate_selectors = [
                'button:has-text("Play")',
                'button:has-text("Generate")',
                'button:has-text("Speak")'
            ]
            
            for selector in generate_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    await self.page.click(selector)
                    print("✅ Generate button clicked")
                    await asyncio.sleep(5)  # Wait for processing
                    return True
                except:
                    continue
            
            print("❌ Could not find generate button")
            return False
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return False
            
    async def download_audio_playwright(self, chunk_index: int) -> str:
        """
        Download audio using Playwright's download API
        This is KEY solution - explicit download path control
        """
        try:
            print(f"⬇️ Starting download for chunk {chunk_index}")
            
            # Clear any existing downloads
            for file in self.download_dir.glob("*"):
                file.unlink()
            
            # Look for download button
            download_selectors = [
                'button:has-text("Download")',
                'button[aria-label*="download"]',
                'a[download]',
                'button:has-text("⬇")'
            ]
            
            download_button = None
            for selector in download_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    download_button = await self.page.query_selector(selector)
                    if download_button:
                        print(f"✅ Found download button: {selector}")
                        break
                except:
                    continue
            
            if not download_button:
                print("❌ Could not find download button")
                return None
            
            # CRITICAL: Start download promise BEFORE clicking
            print("🎯 Setting up download capture...")
            download_promise = self.page.wait_for_event('download')
            
            # Click download button
            print("🖱️ Clicking download button...")
            await download_button.click()
            
            # Wait for download to start
            print("⏳ Waiting for download to start...")
            download = await asyncio.wait_for(download_promise, timeout=30.0)
            
            # Get suggested filename
            suggested_filename = download.suggested_filename()
            print(f"📄 Download started: {suggested_filename}")
            
            # Set our desired filename
            target_filename = f"chunk_{chunk_index:04d}.mp3"
            target_path = self.download_dir / target_filename
            
            # Save to our EXACT desired location
            print(f"💾 Saving to: {target_path}")
            await download.save_as(target_path)
            
            # Wait for download to complete and verify
            await asyncio.sleep(2)
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                print(f"✅ DOWNLOAD SUCCESSFUL!")
                print(f"   📄 File: {target_filename}")
                print(f"   📍 Location: {target_path.absolute()}")
                print(f"   📊 Size: {file_size:,} bytes")
                
                if file_size > 1000:  # At least 1KB
                    return str(target_path)
                else:
                    print(f"⚠️ File too small: {file_size} bytes")
                    return None
            else:
                print(f"❌ File not found at expected location")
                return None
                
        except asyncio.TimeoutError:
            print(f"❌ Download timeout after 30 seconds")
            return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def process_chunk(self, text: str, chunk_index: int, locale: str = "English (South Africa)", voice_gender: str = "male") -> str:
        """Process a single text chunk using Playwright"""
        try:
            # Navigate to site
            print(f"🌐 Navigating to text-to-speech.online...")
            await self.page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
            await asyncio.sleep(3)
            
            # Handle consent
            await self.handle_consent()
            
            # Select locale
            if not await self.select_locale(locale):
                return None
                
            # Select voice
            if not await self.select_voice(voice_gender):
                return None
                
            # Enter text
            if not await self.enter_text(text):
                return None
                
            # Generate speech
            if not await self.generate_speech():
                return None
                
            # Download audio
            downloaded_file = await self.download_audio_playwright(chunk_index)
            
            return downloaded_file
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_index}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            print("🧹 Browser cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def test_playwright_download():
    """Test Playwright download functionality"""
    print("🧪 Testing Playwright Download Solution")
    print("=" * 50)
    
    downloader = PlaywrightTTSDownloader(download_dir="./tts_output")
    
    try:
        # Setup browser
        if not await downloader.setup_browser():
            return False
            
        test_text = "Hello, this is a test of the Playwright download system. We are using English South Africa voice to verify that downloads now work correctly and save to the specified directory."
        
        print(f"📝 Test text: {test_text}")
        
        # Process test chunk
        result = await downloader.process_chunk(
            text=test_text,
            chunk_index=1,
            locale="English (South Africa)",
            voice_gender="male"
        )
        
        if result and Path(result).exists():
            print("\n🎉 PLAYWRIGHT DOWNLOAD TEST SUCCESSFUL!")
            print(f"✅ File saved to: {result}")
            
            # Verify file is in expected directory
            if Path(result).parent == downloader.download_dir:
                print("✅ Download directory: CORRECT")
            else:
                print(f"⚠️ Download directory: UNEXPECTED")
                
            return True
        else:
            print("\n❌ PLAYWRIGHT DOWNLOAD TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await downloader.cleanup()

if __name__ == "__main__":
    # Run test
    success = asyncio.run(test_playwright_download())
    
    if success:
        print("\n" + "=" * 50)
        print("🎯 CONCLUSION: Playwright provides reliable download path control!")
        print("💡 Next step: Implement full Playwright-based TTS automation")
    else:
        print("\n" + "=" * 50)
        print("💥 Further investigation needed")
        
    exit(0 if success else 1)
