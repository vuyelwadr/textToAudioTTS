#!/usr/bin/env python3
"""
Quick test of Playwright download fix
"""

import sys
import os
from pathlib import Path

# Try different Python environments for playwright
python_paths = [
    "/opt/homebrew/bin/python3",
    "/usr/bin/python3", 
    "python3"
]

playwright_found = False
for py_path in python_paths:
    try:
        result = os.system(f"{py_path} -c \"import playwright; print('PLAYWRIGHT WORKS')\" 2>/dev/null")
        if result == 0:
            print(f"✅ Found working Playwright at: {py_path}")
            playwright_found = True
            break
    except:
        continue

if not playwright_found:
    print("❌ No working Playwright installation found")
    print("\nTo install Playwright:")
    print("pip install playwright")
    print("playwright install chromium")
    sys.exit(1)

# Create test script
print("\n🧪 Creating minimal Playwright test...")

test_script = '''
import asyncio
from playwright.async_api import async_playwright

async def test_download():
    print("🎯 Testing Playwright download API...")
    
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context(accept_downloads=True)
    page = await context.new_page()
    
    try:
        # Test navigation
        await page.goto("https://www.text-to-speech.online/")
        print("✅ Navigation successful")
        
        # Test download setup
        download_promise = page.wait_for_event('download', timeout=5000)
        print("✅ Download listener setup successful")
        
        print("🎉 Playwright download API is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()

# Run test
success = asyncio.run(test_download())
print(f"Result: {'SUCCESS' if success else 'FAILED'}")
'''

with open("minimal_playwright_test.py", "w") as f:
    f.write(test_script)

print("✅ Test script created: minimal_playwright_test.py")
print("Run with: python3 minimal_playwright_test.py")
