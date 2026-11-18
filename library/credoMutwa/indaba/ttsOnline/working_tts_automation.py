#!/usr/bin/env python3
"""
Working TTS Online Automation Script
Follows exact workflow: Consent -> Locale -> Voice -> Text -> Download
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium_stealth import stealth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingTTSAutomator:
    def __init__(self, download_dir: str = "./tts_output", headless: bool = False):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.headless = headless
        self.driver = None
        self.session_active = False
        
    def _create_driver(self):
        """Create Chrome driver with proper download configuration"""
        chrome_options = Options()
        
        # Enhanced download preferences - Multiple approaches to ensure downloads work
        prefs = {
            # Core download settings
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "safebrowsing.disable_download_protection": True,
            
            # Content settings to allow automatic downloads
            "profile.default_content_settings.popups": 0,
            "profile.default_content_setting_values.automatic_downloads": 1,
            "profile.content_settings.pattern_pairs.*.multiple-automatic-downloads": 1,
            
            # Additional settings to prevent download prompts
            "profile.default_content_setting_values": {
                "automatic_downloads": 1,
                "popups": 0
            },
            
            # Disable PDF viewer and other plugins that might interfere
            "plugins.always_open_pdf_externally": True,
            
            # Ensure downloads go to specified directory
            "download.default_directory_permissions": "0600"
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Additional arguments for better download handling
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        
        # Headless mode
        if self.headless:
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--window-size=1920,1080")
        
        # Anti-detection options
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")  # Often helps with download issues
        
        # Create driver
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Apply stealth
        stealth(self.driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="MacIntel",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )
        
        # Set page load timeout and download timeout
        self.driver.set_page_load_timeout(30)
        self.driver.set_script_timeout(30)
        
        # Verify download directory was set correctly
        actual_prefs = self.driver.execute_script("""
            return {
                downloadPath: chrome.downloads ? chrome.downloads.defaultDirectory : 'N/A',
                currentDir: navigator.downloads ? navigator.downloads.defaultDirectory : 'N/A'
            };
        """)
        logger.info(f"Chrome driver created successfully")
        logger.info(f"Configured download directory: {self.download_dir.absolute()}")
        logger.info(f"Browser download preferences: {actual_prefs}")
        
    def _wait_for_element(self, locator, timeout=10, description="element"):
        """Wait for element to be present and clickable"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable(locator)
            )
            logger.info(f"Found {description}")
            return element
        except TimeoutException:
            logger.error(f"Could not find {description} within {timeout} seconds")
            return None
    
    def _handle_consent(self):
        """Handle cookie consent dialog"""
        logger.info("Checking for cookie consent...")
        
        # Wait a bit for the consent dialog to fully load
        time.sleep(2)
        
        consent_buttons = [
            "//button[contains(text(), 'Consent')]",
            "//button[contains(text(), 'I agree')]",
            "//button[contains(text(), 'Accept')]",
            "//button[contains(text(), 'Agree')]",
            "//button[contains(text(), 'Accept all')]",
            "//button[@class='btn btn-primary btn-sm acceptcookies']"
        ]
        
        for xpath in consent_buttons:
            try:
                consent_btn = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, xpath))
                )
                
                # Try to click with JavaScript to avoid overlay issues
                self.driver.execute_script("arguments[0].click();", consent_btn)
                logger.info("Cookie consent accepted via JavaScript")
                time.sleep(2)
                return True
                
            except TimeoutException:
                continue
            except Exception as e:
                logger.debug(f"Consent button click failed: {e}")
                continue
        
        # Try clicking by CSS selector as fallback
        consent_css_selectors = [
            "button.acceptcookies",
            "button.btn-primary.acceptcookies",
            ".btn-primary.acceptcookies"
        ]
        
        for css_selector in consent_css_selectors:
            try:
                consent_btn = WebDriverWait(self.driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
                )
                self.driver.execute_script("arguments[0].click();", consent_btn)
                logger.info("Cookie consent accepted via CSS selector")
                time.sleep(2)
                return True
            except:
                continue
        
        logger.info("No consent dialog found or could not click consent")
        return True
    
    def _select_locale(self, locale_name: str = "English (South Africa)"):
        """Select locale by name"""
        logger.info(f"Selecting locale: {locale_name}")
        
        # Try different selectors for locale dropdown
        locale_selectors = [
            "select",
            "select.form-select",
            "select[aria-label*='Locale']",
            "select[aria-label*='Language']"
        ]
        
        locale_element = None
        for selector in locale_selectors:
            locale_element = self._wait_for_element((By.CSS_SELECTOR, selector), 5, f"locale dropdown ({selector})")
            if locale_element:
                break
        
        if not locale_element:
            logger.error("Could not find locale dropdown")
            return False
        
        try:
            select = Select(locale_element)
            
            # Debug: Print available locales
            logger.info("Available locales:")
            for i, option in enumerate(select.options):
                logger.info(f"  {i}: '{option.text}'")
            
            # Find and select the desired locale
            for option in select.options:
                if locale_name in option.text:
                    select.select_by_visible_text(option.text)
                    logger.info(f"Selected locale: {option.text}")
                    time.sleep(2)  # Wait for page to update
                    return True
            
            logger.error(f"Locale '{locale_name}' not found")
            return False
            
        except Exception as e:
            logger.error(f"Error selecting locale: {e}")
            return False
    
    def _select_voice(self, preferred_gender: str = "male"):
        """Select voice by gender preference"""
        logger.info(f"Selecting {preferred_gender} voice")
        
        # Wait for page to update after locale selection
        time.sleep(3)
        
        # Try different selectors for voice dropdown
        voice_selectors = [
            "select:nth-of-type(2)",
            "select.form-select:nth-of-type(2)",
            "select[aria-label*='Voice']",
            "select[aria-label*='voice']"
        ]
        
        voice_element = None
        for selector in voice_selectors:
            try:
                voice_element = self._wait_for_element((By.CSS_SELECTOR, selector), 5, f"voice dropdown ({selector})")
                if voice_element:
                    break
            except:
                continue
        
        if not voice_element:
            # Try finding all selects and use the second one
            try:
                all_selects = self.driver.find_elements(By.TAG_NAME, "select")
                if len(all_selects) >= 2:
                    voice_element = all_selects[1]
                    logger.info("Found voice dropdown as second select element")
            except:
                pass
        
        if not voice_element:
            logger.error("Could not find voice dropdown")
            return False
        
        try:
            select = Select(voice_element)
            
            # Debug: Print available voices
            logger.info("Available voices:")
            for i, option in enumerate(select.options):
                logger.info(f"  {i}: '{option.text}' (value: '{option.get_attribute('value')}')")
            
            # Voice selection patterns for male voices
            male_patterns = ['male', 'guy', 'man', 'adam', 'mark', 'john', 'david']
            female_patterns = ['female', 'woman', 'lady', 'girl', 'sarah', 'jenny', 'leah']
            
            selected_voice = None
            
            if preferred_gender.lower() == "male":
                # Try to find male voice - more direct approach
                logger.info("Looking for male voice...")
                male_voice_found = False
                for option in select.options:
                    option_text = option.text.lower()
                    logger.info(f"Checking option: '{option.text}'")
                    if "(male)" in option_text:
                        select.select_by_visible_text(option.text)
                        selected_voice = option.text
                        male_voice_found = True
                        logger.info(f"✓ Selected male voice: {selected_voice}")
                        break
                
                # If no male voice found, log warning
                if not male_voice_found:
                    logger.warning("No explicit '(male)' voice found!")
                    # Try fallback patterns
                    for option in select.options:
                        option_text = option.text.lower()
                        if any(pattern in option_text for pattern in male_patterns):
                            select.select_by_visible_text(option.text)
                            selected_voice = option.text
                            logger.info(f"⚠ Selected male voice (pattern): {selected_voice}")
                            break
            
            # If no male voice found, try to avoid female voices
            if not selected_voice:
                for option in select.options:
                    option_text = option.text.lower()
                    if not any(pattern in option_text for pattern in female_patterns):
                        select.select_by_visible_text(option.text)
                        selected_voice = option.text
                        logger.info(f"Selected non-female voice: {selected_voice}")
                        break
            
            # If still no selection, use first available
            if not selected_voice and select.options:
                select.select_by_index(0)
                selected_voice = select.options[0].text
                logger.warning(f"Using first available voice: {selected_voice}")
            
            time.sleep(2)  # Wait for voice to load
            return selected_voice is not None
            
        except Exception as e:
            logger.error(f"Error selecting voice: {e}")
            return False
    
    def _enter_text(self, text: str):
        """Enter text into the textarea"""
        logger.info(f"Entering text ({len(text)} characters)")
        
        # Wait for textarea to be available
        textarea_selectors = [
            "textarea",
            "textarea[aria-label*='text']",
            "textarea[placeholder*='text']",
            "#text"
        ]
        
        textarea = None
        for selector in textarea_selectors:
            textarea = self._wait_for_element((By.CSS_SELECTOR, selector), 5, f"textarea ({selector})")
            if textarea:
                break
        
        if not textarea:
            logger.error("Could not find textarea")
            return False
        
        try:
            # Try to focus and click the textarea first
            self.driver.execute_script("arguments[0].focus();", textarea)
            time.sleep(1)
            
            # Clear and enter text
            textarea.clear()
            textarea.send_keys(text)
            logger.info("Text entered successfully")
            time.sleep(2)  # Wait for processing
            return True
        except Exception as e:
            logger.error(f"Error entering text: {e}")
            # Try JavaScript fallback
            try:
                self.driver.execute_script(f"arguments[0].value = '{text}';", textarea)
                time.sleep(2)
                logger.info("Text entered via JavaScript successfully")
                return True
            except Exception as e2:
                logger.error(f"JavaScript text entry also failed: {e2}")
                return False
    
    def _find_downloaded_file_multiple_locations(self, chunk_index: int, max_age_minutes: int = 5):
        """Search for downloaded audio files in multiple possible locations"""
        import os
        from pathlib import Path
        from datetime import datetime, timedelta
        
        # Define possible download locations to check
        possible_locations = [
            self.download_dir,  # Our configured directory
            Path.home() / "Downloads",  # Default Chrome downloads
            Path("/tmp"),  # System temp directory
            Path(os.path.expanduser("~/Desktop")),  # Desktop
            Path.cwd() / "downloads",  # Current working directory
        ]
        
        audio_extensions = ['.wav', '.mp3', '.ogg', '.m4a']
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=max_age_minutes)
        
        logger.info(f"Searching for recently downloaded audio files (modified in last {max_age_minutes} minutes)")
        
        for location in possible_locations:
            if not location.exists():
                logger.debug(f"Location does not exist: {location}")
                continue
                
            logger.debug(f"Checking location: {location}")
            
            try:
                # Search for audio files in this location
                for ext in audio_extensions:
                    pattern = f"*{ext}"
                    for file_path in location.glob(pattern):
                        try:
                            # Check file modification time
                            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                            
                            # Only consider files modified recently
                            if file_mtime > cutoff_time:
                                file_size = file_path.stat().st_size
                                
                                # Log the found file
                                logger.info(f"Found recent audio file: {file_path}")
                                logger.info(f"  Location: {location}")
                                logger.info(f"  Size: {file_size} bytes")
                                logger.info(f"  Modified: {file_mtime}")
                                logger.info(f"  Age: {(current_time - file_mtime).total_seconds():.1f} seconds")
                                
                                # Only return if file is reasonably sized
                                if file_size > 1000:  # At least 1KB
                                    logger.info(f"✓ Valid audio file found")
                                    return file_path
                                else:
                                    logger.warning(f"⚠ File too small: {file_size} bytes")
                                    
                        except (OSError, IOError) as e:
                            logger.debug(f"Error accessing file {file_path}: {e}")
                            continue
                            
            except Exception as e:
                logger.debug(f"Error searching location {location}: {e}")
                continue
        
        logger.info("No recent valid audio files found in any location")
        return None

    def _download_audio(self, chunk_index: int):
        """Click download button and wait for file"""
        logger.info(f"Initiating download for chunk {chunk_index}")
        
        # Clear download directory
        for file in self.download_dir.glob("*.wav"):
            file.unlink()
        for file in self.download_dir.glob("*.mp3"):
            file.unlink()
        
        # Wait for processing to complete
        logger.info("Waiting for audio processing...")
        time.sleep(5)
        
        # Look for download button
        download_selectors = [
            "button",
            "button:contains('Download')",
            "button[aria-label*='download']",
            "a[download]",
            ".download-btn",
            "button span:contains('Download')"
        ]
        
        download_element = None
        
        # First try specific selectors
        for selector in ["button[aria-label*='download']", "a[download]"]:
            try:
                download_element = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                if download_element:
                    logger.info(f"Found download button: {selector}")
                    break
            except:
                continue
        
        # If not found, search by text content
        if not download_element:
            try:
                # Try XPath to find button with Download text
                download_element = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Download')]")
                logger.info("Found download button by text content")
            except:
                pass
        
        # If still not found, try any button
        if not download_element:
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if "download" in button.text.lower():
                    download_element = button
                    logger.info(f"Found download button with text: {button.text}")
                    break
        
        if not download_element:
            # Take screenshot for debugging
            screenshot_path = self.download_dir / f"debug_screenshot_{chunk_index}.png"
            self.driver.save_screenshot(str(screenshot_path))
            logger.error(f"Could not find download button. Screenshot saved to {screenshot_path}")
            return None
        
        try:
            # Record time before download
            download_start_time = time.time()
            
            # Enhanced debugging: Get download button details
            button_info = self.driver.execute_script("""
                var element = arguments[0];
                return {
                    tagName: element.tagName,
                    href: element.href,
                    download: element.download,
                    onclick: element.onclick ? element.onclick.toString() : 'none',
                    innerHTML: element.innerHTML,
                    outerHTML: element.outerHTML.substring(0, 200) + '...'
                };
            """, download_element)
            logger.info(f"Download button details: {button_info}")
            
            # Check if this might be triggering a JavaScript download
            page_url = self.driver.current_url
            logger.info(f"Current page URL: {page_url}")
            
            # Try to click download button with JavaScript to avoid overlay issues
            try:
                # Monitor for new windows/tabs that might open
                original_handles = self.driver.window_handles
                
                self.driver.execute_script("arguments[0].click();", download_element)
                logger.info("Download button clicked via JavaScript")
                
                # Check if new window/tab opened
                time.sleep(1)
                new_handles = self.driver.window_handles
                if len(new_handles) > len(original_handles):
                    logger.info(f"New window/tab detected for download")
                    for handle in new_handles:
                        if handle not in original_handles:
                            self.driver.switch_to.window(handle)
                            new_url = self.driver.current_url
                            logger.info(f"New window URL: {new_url}")
                            # This might be a direct download link
                            if new_url.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                                logger.info(f"Direct download URL detected: {new_url}")
                            self.driver.close()
                    self.driver.switch_to.window(original_handles[0])
                    
            except Exception as e:
                logger.debug(f"JavaScript click failed, trying regular click: {e}")
                download_element.click()
                logger.info("Download button clicked via regular click")
            
            # Wait for download to complete with enhanced monitoring
            logger.info("Waiting for download to complete...")
            max_wait = 30
            download_complete = False
            downloaded_file = None
            
            for i in range(max_wait):
                elapsed_time = time.time() - download_start_time
                
                # First, check our configured directory
                local_files = []
                for ext in ['*.wav', '*.mp3', '*.ogg', '*.m4a']:
                    local_files.extend(self.download_dir.glob(ext))
                
                if local_files:
                    logger.info(f"Found {len(local_files)} files in configured directory")
                    for local_file in local_files:
                        file_size = local_file.stat().st_size
                        file_mtime = local_file.stat().st_mtime
                        
                        # Check if file was created after download started
                        if file_mtime > download_start_time and file_size > 1000:
                            logger.info(f"✓ Downloaded to configured directory: {local_file.name} ({file_size} bytes)")
                            downloaded_file = local_file
                            download_complete = True
                            break
                
                # If not found in our directory, search all possible locations
                if not download_complete and i >= 5:  # Start searching after 5 seconds
                    logger.info(f"Searching all possible download locations (elapsed: {elapsed_time:.1f}s)")
                    found_file = self._find_downloaded_file_multiple_locations(chunk_index, max_age_minutes=1)
                    
                    if found_file:
                        downloaded_file = found_file
                        download_complete = True
                        
                        # If file is not in our directory, try to move it
                        if found_file.parent != self.download_dir:
                            try:
                                destination = self.download_dir / f"chunk_{chunk_index:03d}{found_file.suffix}"
                                found_file.rename(destination)
                                downloaded_file = destination
                                logger.info(f"✓ Moved downloaded file to: {destination}")
                            except Exception as e:
                                logger.warning(f"Could not move file from {found_file} to {destination}: {e}")
                                # Keep original location if move fails
                        break
                
                if download_complete:
                    break
                
                # Progress update every 5 seconds
                if i % 5 == 0:
                    logger.info(f"Download progress: {i+1}s elapsed, still waiting...")
                    
                    # Debug: show contents of our download directory
                    dir_contents = list(self.download_dir.glob("*"))
                    if dir_contents:
                        logger.info(f"Download directory contents:")
                        for item in dir_contents:
                            if item.is_file():
                                file_size = item.stat().st_size
                                file_age = time.time() - item.stat().st_mtime
                                logger.info(f"  {item.name} ({file_size} bytes, age: {file_age:.1f}s)")
                    else:
                        logger.info("Download directory is empty")
                
                time.sleep(1)
            
            if download_complete and downloaded_file:
                # Final verification
                file_size = downloaded_file.stat().st_size
                if file_size > 1000:
                    # Ensure file has proper naming
                    if not downloaded_file.name.startswith(f"chunk_{chunk_index:03d}"):
                        new_name = self.download_dir / f"chunk_{chunk_index:03d}{downloaded_file.suffix}"
                        try:
                            downloaded_file.rename(new_name)
                            downloaded_file = new_name
                        except Exception as e:
                            logger.warning(f"Could not rename file: {e}")
                    
                    logger.info(f"✓ DOWNLOAD SUCCESSFUL: {downloaded_file} ({file_size} bytes)")
                    return str(downloaded_file)
                else:
                    logger.error(f"Downloaded file too small: {file_size} bytes")
                    return None
            else:
                logger.error(f"Download failed - no valid file found after {max_wait} seconds")
                
                # Final comprehensive search
                logger.info("Performing final comprehensive search for downloaded files...")
                final_search = self._find_downloaded_file_multiple_locations(chunk_index, max_age_minutes=2)
                if final_search:
                    logger.info(f"Found file in final search: {final_search}")
                    return str(final_search)
                
                return None
            
        except Exception as e:
            logger.error(f"Error during download: {e}")
            return None
    
    def start_session(self):
        """Start browser session"""
        if self.session_active:
            logger.warning("Session already active")
            return True
        
        try:
            self._create_driver()
            self.driver.get("https://www.text-to-speech.online/")
            time.sleep(3)
            
            # Handle consent
            self._handle_consent()
            
            self.session_active = True
            logger.info("Session started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False
    
    def process_chunk(self, text: str, chunk_index: int, locale: str = "English (South Africa)", voice_gender: str = "male") -> Optional[str]:
        """Process a single text chunk"""
        if not self.session_active:
            if not self.start_session():
                return None
        
        try:
            # Select locale
            if not self._select_locale(locale):
                return None
            
            # Select voice
            if not self._select_voice(voice_gender):
                return None
            
            # Enter text
            if not self._enter_text(text):
                return None
            
            # Download audio
            downloaded_file = self._download_audio(chunk_index)
            
            return downloaded_file
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {e}")
            return None
    
    def end_session(self):
        """End browser session"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        self.session_active = False
        logger.info("Session ended")

def test_single_chunk():
    """Test processing a single chunk with enhanced download verification"""
    automator = WorkingTTSAutomator(headless=False)  # Use visible browser for testing
    
    try:
        test_text = "Hello, this is a test of the text to speech system. We are using English South Africa voice to generate this audio file."
        
        logger.info(f"🧪 Testing Chrome download fix...")
        logger.info(f"📁 Configured download directory: {automator.download_dir.absolute()}")
        logger.info(f"📝 Test text length: {len(test_text)} characters")
        
        result = automator.process_chunk(
            text=test_text,
            chunk_index=1,
            locale="English (South Africa)",
            voice_gender="male"
        )
        
        if result:
            import os
            file_path = Path(result)
            file_size = file_path.stat().st_size
            
            logger.info(f"✅ SUCCESS: Chrome download fix working!")
            logger.info(f"   📄 File: {file_path.name}")
            logger.info(f"   📍 Location: {file_path.absolute()}")
            logger.info(f"   📊 Size: {file_size:,} bytes")
            
            # Verify file is in the expected directory
            if file_path.parent == automator.download_dir:
                logger.info(f"   ✅ Download directory: CORRECT")
            else:
                logger.warning(f"   ⚠️ Download directory: UNEXPECTED (wanted: {automator.download_dir})")
                
            return True
        else:
            logger.error("✗ FAILED: No audio file generated")
            return False
            
    finally:
        automator.end_session()

if __name__ == "__main__":
    success = test_single_chunk()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
