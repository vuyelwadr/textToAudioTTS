#!/usr/bin/env python3
"""
TTS Online Automation Script
Automates text-to-speech generation using text-to-speech.online
Based on existing chunked TTS patterns in the codebase
"""

import os
import sys
import time
import logging
import json
import re
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Generator, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import psutil
from tqdm import tqdm

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import (
        TimeoutException, 
        NoSuchElementException, 
        WebDriverException,
        ElementNotInteractableException
    )
    from selenium_stealth import stealth
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install selenium selenium-stealth")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
WEBSITE_URL = "https://www.text-to-speech.online/"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEFAULT_OUTPUT_DIR = "tts_online_output"
CHUNK_SIZE = 5000  # characters per chunk (safe margin under 6000 limit)
DEFAULT_LOCALE = "English (South Africa)"
DEFAULT_VOICE = "Microsoft Guy Online (Natural)"  # Male voice
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 1.0
MAX_CONCURRENT_BROWSERS = 3
REQUEST_TIMEOUT = 30  # seconds
ELEMENT_WAIT_TIMEOUT = 10  # seconds

class DualOutputLogger:
    """Custom logger that outputs to both terminal and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

class ProgressTracker:
    """Enhanced progress tracker with time estimation and memory monitoring"""
    def __init__(self, total_items):
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.item_times = []
        self.last_update_time = time.time()
        self.lock = threading.Lock()

    def update(self, items_done=1, success=True):
        with self.lock:
            if success:
                self.processed_items += items_done
            else:
                self.failed_items += items_done
            
            current_time = time.time()
            if self.last_update_time:
                batch_time = current_time - self.last_update_time
                self.item_times.append(batch_time / items_done)
                if len(self.item_times) > 20:
                    self.item_times.pop(0)
            
            self.last_update_time = current_time

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        remaining_items = self.total_items - self.processed_items - self.failed_items
        percentage = ((self.processed_items + self.failed_items) / self.total_items * 100) if self.total_items > 0 else 0
        
        if self.item_times and remaining_items > 0:
            avg_time_per_item = sum(self.item_times) / len(self.item_times)
            eta_seconds = remaining_items * avg_time_per_item
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "Calculating..."
        
        return {
            'processed': self.processed_items,
            'failed': self.failed_items,
            'total': self.total_items,
            'remaining': remaining_items,
            'percentage': percentage,
            'elapsed': str(timedelta(seconds=int(elapsed_time))),
            'eta': str(eta) if isinstance(eta, timedelta) else eta,
            'memory_percent': psutil.virtual_memory().percent,
            'items_per_second': self.processed_items / elapsed_time if elapsed_time > 0 else 0
        }

class TTSOnlineAutomator:
    """Automates TTS generation using text-to-speech.online"""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, headless: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headless = headless
        self.driver = None
        self.session_stats = {
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_characters': 0,
            'start_time': None,
            'end_time': None
        }
        
    def _setup_chrome_driver(self) -> webdriver.Chrome:
        """Setup Chrome WebDriver with stealth options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        # Anti-detection options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Performance optimizations
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        # Set realistic user agent
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ]
        chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
        
        # Create driver
        try:
            driver = webdriver.Chrome(options=chrome_options)
            
            # Apply stealth
            stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
            )
            
            # Set page load timeout
            driver.set_page_load_timeout(REQUEST_TIMEOUT)
            driver.implicitly_wait(5)
            
            return driver
            
        except Exception as e:
            logger.error(f"Failed to create Chrome driver: {e}")
            raise

    def _navigate_to_site(self) -> bool:
        """Navigate to the TTS website"""
        try:
            logger.info(f"Navigating to {WEBSITE_URL}")
            self.driver.get(WEBSITE_URL)
            
            # Wait for page to load
            WebDriverWait(self.driver, ELEMENT_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Random delay to seem more human
            time.sleep(random.uniform(1, 3))
            
            return True
            
        except TimeoutException:
            logger.error("Timeout while navigating to website")
            return False
        except WebDriverException as e:
            logger.error(f"WebDriver error while navigating: {e}")
            return False

    def _select_locale(self, locale: str = DEFAULT_LOCALE) -> bool:
        """Select the desired locale"""
        try:
            logger.info(f"Selecting locale: {locale}")
            
            # Look for locale dropdown (may need adjustment based on actual site structure)
            locale_selectors = [
                "select[aria-label*='locale']",
                "select[aria-label*='language']",
                "#locale",
                "#language",
                "select.form-select"
            ]
            
            locale_element = None
            for selector in locale_selectors:
                try:
                    locale_element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not locale_element:
                logger.warning("Could not find locale selector, using default")
                return True
            
            # Select the desired locale
            from selenium.webdriver.support.ui import Select
            select = Select(locale_element)
            
            # Try to find the option
            for option in select.options:
                if locale.lower() in option.text.lower():
                    select.select_by_visible_text(option.text)
                    logger.info(f"Selected locale: {option.text}")
                    time.sleep(random.uniform(0.5, 1.5))
                    return True
            
            logger.warning(f"Locale '{locale}' not found, using default")
            return True
            
        except Exception as e:
            logger.error(f"Error selecting locale: {e}")
            return False

    def _select_voice(self, voice: str = DEFAULT_VOICE) -> bool:
        """Select the desired voice"""
        try:
            logger.info(f"Selecting voice: {voice}")
            
            # Look for voice dropdown
            voice_selectors = [
                "select[aria-label*='voice']",
                "#voice",
                "select.form-select:nth-child(2)"
            ]
            
            voice_element = None
            for selector in voice_selectors:
                try:
                    voice_element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not voice_element:
                logger.warning("Could not find voice selector, using default")
                return True
            
            # Select the desired voice
            from selenium.webdriver.support.ui import Select
            select = Select(voice_element)
            
            # Debug: Print all available voices
            logger.info("Available voices:")
            for i, option in enumerate(select.options):
                logger.info(f"  {i}: '{option.text}' (value: '{option.get_attribute('value')}')")
            
            # Try to find a male voice
            selected_voice = None
            for option in select.options:
                option_text = option.text.lower()
                if "male" in option_text or "guy" in option_text or "man" in option_text:
                    select.select_by_visible_text(option.text)
                    selected_voice = option.text
                    logger.info(f"Selected male voice: {option.text}")
                    time.sleep(random.uniform(0.5, 1.5))
                    return True
            
            # If no male voice found, try the specified voice
            for option in select.options:
                if voice.lower() in option.text.lower():
                    select.select_by_visible_text(option.text)
                    selected_voice = option.text
                    logger.info(f"Selected voice: {option.text}")
                    time.sleep(random.uniform(0.5, 1.5))
                    return True
            
            logger.warning(f"Voice '{voice}' not found, using default")
            return True
            
        except Exception as e:
            logger.error(f"Error selecting voice: {e}")
            return False

    def _set_speed_pitch(self, speed: float = DEFAULT_SPEED, pitch: float = DEFAULT_PITCH) -> bool:
        """Set speed and pitch controls"""
        try:
            logger.info(f"Setting speed: {speed}, pitch: {pitch}")
            
            # Look for speed control
            speed_selectors = [
                "input[type='range'][aria-label*='speed']",
                "input[type='range'][name*='speed']",
                "#speed"
            ]
            
            for selector in speed_selectors:
                try:
                    speed_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    self.driver.execute_script(f"arguments[0].value = {speed};", speed_element)
                    break
                except NoSuchElementException:
                    continue
            
            # Look for pitch control
            pitch_selectors = [
                "input[type='range'][aria-label*='pitch']",
                "input[type='range'][name*='pitch']",
                "#pitch"
            ]
            
            for selector in pitch_selectors:
                try:
                    pitch_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    self.driver.execute_script(f"arguments[0].value = {pitch};", pitch_element)
                    break
                except NoSuchElementException:
                    continue
            
            time.sleep(random.uniform(0.5, 1))
            return True
            
        except Exception as e:
            logger.error(f"Error setting speed/pitch: {e}")
            return False

    def _input_text(self, text: str) -> bool:
        """Input text into the text area"""
        try:
            logger.info(f"Inputting text ({len(text)} characters)")
            
            # Look for text area
            text_selectors = [
                "textarea[aria-label*='text']",
                "textarea[placeholder*='text']",
                "#text",
                "textarea.form-control"
            ]
            
            text_element = None
            for selector in text_selectors:
                try:
                    text_element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not text_element:
                logger.error("Could not find text input area")
                return False
            
            # Clear and input text
            text_element.clear()
            
            # Type text with human-like delays
            for char in text:
                text_element.send_keys(char)
                if random.random() < 0.1:  # Occasional pause
                    time.sleep(random.uniform(0.05, 0.15))
            
            time.sleep(random.uniform(1, 2))
            return True
            
        except Exception as e:
            logger.error(f"Error inputting text: {e}")
            return False

    def _generate_speech(self) -> bool:
        """Click generate button and wait for completion"""
        try:
            logger.info("Generating speech...")
            
            # Look for generate button
            generate_selectors = [
                "button[aria-label*='generate']",
                "button[aria-label*='speak']",
                "button:contains('Generate')",
                "button:contains('Speak')",
                "button.btn-primary"
            ]
            
            generate_button = None
            for selector in generate_selectors:
                try:
                    if ":contains(" in selector:
                        # Handle jQuery-style selectors
                        text = selector.split("'")[1]
                        elements = self.driver.find_elements(By.TAG_NAME, "button")
                        for elem in elements:
                            if text.lower() in elem.text.lower():
                                generate_button = elem
                                break
                    else:
                        generate_button = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    
                    if generate_button:
                        break
                        
                except (TimeoutException, NoSuchElementException):
                    continue
            
            if not generate_button:
                logger.error("Could not find generate button")
                return False
            
            # Click generate button
            generate_button.click()
            time.sleep(random.uniform(1, 2))
            
            # Wait for generation to complete (look for download button or audio element)
            completion_selectors = [
                "button[aria-label*='download']",
                "a[download]",
                "audio[src]",
                ".download-btn"
            ]
            
            for selector in completion_selectors:
                try:
                    WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info("Speech generation completed")
                    time.sleep(random.uniform(1, 2))
                    return True
                except TimeoutException:
                    continue
            
            logger.warning("Could not confirm speech generation completion")
            return True  # Assume it worked anyway
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False

    def _download_audio(self, chunk_index: int, text_hash: str) -> Optional[str]:
        """Download the generated audio file"""
        try:
            logger.info(f"Downloading audio for chunk {chunk_index}")
            
            # Look for download button
            download_selectors = [
                "button[aria-label*='download']",
                "a[download]",
                ".download-btn"
            ]
            
            download_element = None
            for selector in download_selectors:
                try:
                    download_element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue
            
            if not download_element:
                logger.error("Could not find download button")
                return None
            
            # Set download path
            filename = f"chunk_{chunk_index:04d}_{text_hash}.mp3"
            filepath = self.output_dir / filename
            
            # Handle the download
            if download_element.tag_name == "a":
                download_url = download_element.get_attribute("href")
                if download_url:
                    import requests
                    response = requests.get(download_url)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded: {filename}")
                    return str(filepath)
            else:
                # For button clicks, handle Chrome download
                chrome_options = self.driver.options
                prefs = {
                    "download.default_directory": str(self.output_dir.absolute()),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": True
                }
                chrome_options.add_experimental_option("prefs", prefs)
                
                download_element.click()
                time.sleep(3)  # Wait for download
                
                # Check if file was downloaded
                downloaded_files = list(self.output_dir.glob("*.mp3"))
                if downloaded_files:
                    latest_file = max(downloaded_files, key=os.path.getctime)
                    # Rename to our desired filename
                    new_path = self.output_dir / filename
                    latest_file.rename(new_path)
                    logger.info(f"Downloaded: {filename}")
                    return str(new_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None

    def _process_chunk(self, chunk_text: str, chunk_index: int, retry_count: int = 0) -> Optional[str]:
        """Process a single text chunk"""
        try:
            # Create text hash for filename
            text_hash = abs(hash(chunk_text[:100])) % 10000
            
            # Check if file already exists
            filename = f"chunk_{chunk_index:04d}_{text_hash}.mp3"
            filepath = self.output_dir / filename
            if filepath.exists():
                logger.info(f"Skipping existing chunk {chunk_index}")
                return str(filepath)
            
            # Setup driver for this chunk
            if not self.driver:
                self.driver = self._setup_chrome_driver()
            
            # Navigate and configure
            if not self._navigate_to_site():
                raise Exception("Failed to navigate to website")
            
            if not self._select_locale():
                logger.warning("Failed to select locale, continuing with default")
            
            if not self._select_voice():
                logger.warning("Failed to select voice, continuing with default")
            
            if not self._set_speed_pitch():
                logger.warning("Failed to set speed/pitch, continuing with defaults")
            
            if not self._input_text(chunk_text):
                raise Exception("Failed to input text")
            
            if not self._generate_speech():
                raise Exception("Failed to generate speech")
            
            downloaded_file = self._download_audio(chunk_index, text_hash)
            if not downloaded_file:
                raise Exception("Failed to download audio")
            
            return downloaded_file
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {e}")
            
            # Retry logic
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying chunk {chunk_index} (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY * (retry_count + 1))
                
                # Reset driver
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                
                return self._process_chunk(chunk_text, chunk_index, retry_count + 1)
            else:
                logger.error(f"Failed to process chunk {chunk_index} after {MAX_RETRIES} retries")
                return None

    def _split_text_into_chunks(self, text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
        """Split text into intelligible chunks"""
        logger.info(f"Splitting text into chunks (max {chunk_size} characters each)")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (sentence + " ").strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single sentence is too long, split it
                if len(sentence) > chunk_size:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= chunk_size:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk:
                        current_chunk = temp_chunk.strip()
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def process_text_file(self, input_file: str, start_chunk: int = 0, end_chunk: Optional[int] = None) -> Dict[str, Any]:
        """Process a text file and generate audio for all chunks"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Processing text file: {input_file}")
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks
        chunks = self._split_text_into_chunks(text)
        
        if end_chunk is None:
            end_chunk = len(chunks)
        
        # Process chunk range
        chunks_to_process = chunks[start_chunk:end_chunk]
        
        # Setup progress tracking
        progress = ProgressTracker(len(chunks_to_process))
        
        # Setup logging
        log_filename = self.output_dir / f"tts_online_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        dual_logger = DualOutputLogger(str(log_filename))
        sys.stdout = dual_logger
        sys.stderr = dual_logger
        
        # Update session stats
        self.session_stats['total_chunks'] = len(chunks_to_process)
        self.session_stats['total_characters'] = sum(len(chunk) for chunk in chunks_to_process)
        self.session_stats['start_time'] = datetime.now()
        
        # Process chunks
        successful_files = []
        failed_chunks = []
        
        try:
            with tqdm(total=len(chunks_to_process), desc="Processing chunks", unit="chunk") as pbar:
                for i, chunk in enumerate(chunks_to_process):
                    chunk_index = start_chunk + i
                    
                    try:
                        logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks_to_process)}")
                        
                        # Process the chunk
                        result_file = self._process_chunk(chunk, chunk_index)
                        
                        if result_file:
                            successful_files.append(result_file)
                            progress.update(1, success=True)
                            logger.info(f"✓ Successfully processed chunk {chunk_index}")
                        else:
                            failed_chunks.append(chunk_index)
                            progress.update(1, success=False)
                            logger.error(f"✗ Failed to process chunk {chunk_index}")
                        
                        # Update progress bar
                        stats = progress.get_stats()
                        pbar.set_postfix({
                            'Success': f"{progress.processed_items}",
                            'Failed': f"{progress.failed_items}",
                            'Memory': f"{stats['memory_percent']:.1f}%",
                            'Speed': f"{stats['items_per_second']:.2f} chunks/s"
                        })
                        pbar.update(1)
                        
                        # Add delay between chunks to avoid rate limiting
                        time.sleep(random.uniform(2, 5))
                        
                    except KeyboardInterrupt:
                        logger.info("Processing interrupted by user")
                        break
                    except Exception as e:
                        logger.error(f"Unexpected error processing chunk {chunk_index}: {e}")
                        failed_chunks.append(chunk_index)
                        progress.update(1, success=False)
                        pbar.update(1)
                        continue
        
        finally:
            # Cleanup
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            
            # Restore stdout/stderr
            sys.stdout = dual_logger.terminal
            sys.stderr = dual_logger.terminal
            dual_logger.close()
        
        # Update final stats
        self.session_stats['successful_chunks'] = len(successful_files)
        self.session_stats['failed_chunks'] = len(failed_chunks)
        self.session_stats['end_time'] = datetime.now()
        
        # Save session report
        self._save_session_report(successful_files, failed_chunks)
        
        return {
            'successful_files': successful_files,
            'failed_chunks': failed_chunks,
            'total_chunks': len(chunks_to_process),
            'success_rate': len(successful_files) / len(chunks_to_process) * 100,
            'session_stats': self.session_stats
        }

    def _save_session_report(self, successful_files: List[str], failed_chunks: List[int]):
        """Save a detailed session report"""
        report = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'website': WEBSITE_URL,
                'output_directory': str(self.output_dir),
                'settings': {
                    'locale': DEFAULT_LOCALE,
                    'voice': DEFAULT_VOICE,
                    'speed': DEFAULT_SPEED,
                    'pitch': DEFAULT_PITCH,
                    'chunk_size': CHUNK_SIZE,
                    'max_retries': MAX_RETRIES
                }
            },
            'statistics': self.session_stats,
            'results': {
                'successful_files': successful_files,
                'failed_chunks': failed_chunks,
                'success_rate': len(successful_files) / self.session_stats['total_chunks'] * 100 if self.session_stats['total_chunks'] > 0 else 0
            }
        }
        
        report_file = self.output_dir / f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session report saved: {report_file}")

def main():
    """Main function for command line usage"""
    global CHUNK_SIZE
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Online Automation Script")
    parser.add_argument("input_file", help="Input text file to process")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start-chunk", type=int, default=0, help="Start chunk index")
    parser.add_argument("--end-chunk", type=int, help="End chunk index")
    parser.add_argument("--no-headless", action="store_true", help="Disable headless mode")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    
    args = parser.parse_args()
    
    # Create automator
    automator = TTSOnlineAutomator(
        output_dir=args.output,
        headless=not args.no_headless
    )
    
    # Override chunk size if specified
    CHUNK_SIZE = args.chunk_size
    
    try:
        # Process the file
        result = automator.process_text_file(
            args.input_file,
            start_chunk=args.start_chunk,
            end_chunk=args.end_chunk
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Successful: {result['successful_files']}")
        print(f"Failed: {len(result['failed_chunks'])}")
        print(f"Success rate: {result['success_rate']:.1f}%")
        print(f"Output directory: {automator.output_dir}")
        
        if result['failed_chunks']:
            print(f"\nFailed chunks: {result['failed_chunks']}")
            print("You can retry these chunks using --start-chunk and --end-chunk")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
