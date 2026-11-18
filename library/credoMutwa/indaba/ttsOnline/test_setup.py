#!/usr/bin/env python3
"""
Test script to verify TTS Online automation setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        from selenium import webdriver
        print("✓ selenium imported successfully")
    except ImportError as e:
        print(f"✗ selenium import failed: {e}")
        return False
    
    try:
        from selenium_stealth import stealth
        print("✓ selenium-stealth imported successfully")
    except ImportError as e:
        print(f"✗ selenium-stealth import failed: {e}")
        return False
    
    try:
        import requests
        print("✓ requests imported successfully")
    except ImportError as e:
        print(f"✗ requests import failed: {e}")
        return False
    
    try:
        import psutil
        print("✓ psutil imported successfully")
    except ImportError as e:
        print(f"✗ psutil import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False
    
    return True

def test_chrome_driver():
    """Test if Chrome WebDriver can be created"""
    print("\nTesting Chrome WebDriver...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.google.com")
        
        title = driver.title
        print(f"✓ Chrome WebDriver working - Page title: {title}")
        
        driver.quit()
        return True
        
    except Exception as e:
        print(f"✗ Chrome WebDriver test failed: {e}")
        print("\nPossible solutions:")
        print("1. Install Chrome browser")
        print("2. Install ChromeDriver:")
        print("   - On macOS: brew install chromedriver")
        print("   - Or download from: https://chromedriver.chromium.org/")
        print("3. Make sure ChromeDriver is in your PATH")
        return False

def test_automation_script():
    """Test if the automation script can be imported"""
    print("\nTesting automation script import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from ttsOnline_automation import TTSOnlineAutomator
        print("✓ TTSOnlineAutomator imported successfully")
        
        # Test instantiation
        automator = TTSOnlineAutomator(headless=True)
        print("✓ TTSOnlineAutomator instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Automation script test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TTS ONLINE AUTOMATION SETUP TEST")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test Chrome driver
    if not test_chrome_driver():
        all_passed = False
    
    # Test automation script
    if not test_automation_script():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Setup is ready!")
        print("\nUsage examples:")
        print("1. Process entire file:")
        print("   python ttsOnline_automation.py indaba1.1.txt")
        print("\n2. Process chunk range:")
        print("   python ttsOnline_automation.py indaba1.1.txt --start-chunk 0 --end-chunk 10")
        print("\n3. Run with visible browser (for debugging):")
        print("   python ttsOnline_automation.py indaba1.1.txt --no-headless")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before using")
        print("\nInstall missing dependencies:")
        print("pip install -r requirements.txt")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
