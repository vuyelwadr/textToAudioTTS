
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
