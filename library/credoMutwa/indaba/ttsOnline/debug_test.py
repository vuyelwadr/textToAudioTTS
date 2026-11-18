#!/usr/bin/env python3
"""
Debug test for TTS automation - focused on getting one successful download
"""

import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_driver(download_dir):
    """Create Chrome driver with download configuration"""
    chrome_options = Options()
    
    # Set up download directory
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Other options
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Create driver
    driver = webdriver.Chrome(options=chrome_options)
    
    # Apply stealth
    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="MacIntel",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    
    return driver

def test_tts_download():
    """Test the complete TTS workflow with actual download"""
    
    # Create download directory
    download_dir = "./debug_download"
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        # Create driver
        driver = create_driver(download_dir)
        
        # Navigate to site
        logger.info("Navigating to TTS website...")
        driver.get("https://www.text-to-speech.online/")
        time.sleep(3)
        
        # Handle cookie consent if present
        try:
            consent_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Consent')]"))
            )
            consent_button.click()
            logger.info("Accepted cookie consent")
            time.sleep(2)
        except:
            logger.info("No cookie consent found")
        
        # Select English South Africa locale
        logger.info("Selecting English South Africa locale...")
        locale_select = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "select"))
        )
        select = Select(locale_select)
        
        # Debug: Print all available locales
        logger.info("Available locales:")
        for option in select.options:
            logger.info(f"  '{option.text}'")
        
        # Select English South Africa
        for option in select.options:
            if "English (South Africa)" in option.text:
                select.select_by_visible_text(option.text)
                logger.info(f"Selected locale: {option.text}")
                break
        
        time.sleep(2)
        
        # Select voice
        logger.info("Looking for voice selector...")
        voice_select = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "select:nth-of-type(2)"))
        )
        voice_select_obj = Select(voice_select)
        
        # Debug: Print all available voices
        logger.info("Available voices:")
        for option in voice_select_obj.options:
            logger.info(f"  '{option.text}' (value: '{option.get_attribute('value')}')")
        
        # Try to select male voice
        male_voice_selected = False
        for option in voice_select_obj.options:
            option_text = option.text.lower()
            if any(male in option_text for male in ['male', 'guy', 'man']):
                voice_select_obj.select_by_visible_text(option.text)
                logger.info(f"Selected male voice: {option.text}")
                male_voice_selected = True
                break
        
        if not male_voice_selected:
            logger.warning("No male voice found, using first available voice")
            voice_select_obj.select_by_index(0)
        
        time.sleep(2)
        
        # Enter test text
        logger.info("Entering test text...")
        textarea = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )
        test_text = "Hello, this is a test of the text to speech system. We are using English South Africa male voice to generate this audio file."
        textarea.clear()
        textarea.send_keys(test_text)
        time.sleep(2)
        
        # Wait for processing
        logger.info("Waiting for text processing...")
        time.sleep(5)
        
        # Look for download button
        logger.info("Looking for download button...")
        download_selectors = [
            "button:contains('Download')",
            "button[aria-label*='download']",
            "a[download]",
            ".download-btn",
            "button span:contains('Download')"
        ]
        
        download_element = None
        for selector in download_selectors:
            try:
                if "contains" in selector:
                    # Handle XPath contains - skip for now, use simpler approach
                    continue
                else:
                    download_element = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                logger.info(f"Found download element with selector: {selector}")
                break
            except:
                continue
        
        if not download_element:
            # Try to find any button with Download text
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if "download" in button.text.lower():
                    download_element = button
                    logger.info(f"Found download button by text: {button.text}")
                    break
        
        if not download_element:
            # Take screenshot for debugging
            screenshot_path = os.path.join(download_dir, "debug_screenshot.png")
            driver.save_screenshot(screenshot_path)
            logger.error(f"Could not find download button. Screenshot saved to {screenshot_path}")
            
            # Also save page source
            with open(os.path.join(download_dir, "debug_page.html"), "w") as f:
                f.write(driver.page_source)
            
            return False
        
        # Clear download directory before download
        for file in os.listdir(download_dir):
            if file.endswith(('.mp3', '.wav')):
                os.remove(os.path.join(download_dir, file))
        
        # Click download
        logger.info("Clicking download button...")
        download_element.click()
        
        # Wait for download
        logger.info("Waiting for download to complete...")
        max_wait = 30
        for i in range(max_wait):
            downloaded_files = [f for f in os.listdir(download_dir) if f.endswith(('.mp3', '.wav'))]
            if downloaded_files:
                downloaded_file = os.path.join(download_dir, downloaded_files[0])
                file_size = os.path.getsize(downloaded_file)
                logger.info(f"Downloaded file: {downloaded_files[0]} ({file_size} bytes)")
                return True
            time.sleep(1)
        
        logger.error("Download failed - no file found after 30 seconds")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    finally:
        # Clean up
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    success = test_tts_download()
    if success:
        logger.info("✓ Test successful - audio file downloaded")
    else:
        logger.error("✗ Test failed - no audio file downloaded")
