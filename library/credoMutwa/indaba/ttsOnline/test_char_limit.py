#!/usr/bin/env python3
"""Test to find the actual character limit for TTS website"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def test_char_limit(text_length):
    """Test if a specific text length works"""

    test_text = "a " * (text_length // 2)  # Simple repeated text
    test_text = test_text[:text_length]

    print(f"\n{'='*70}")
    print(f"Testing {text_length} characters...")
    print(f"{'='*70}")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(accept_downloads=True)
    page = await context.new_page()

    try:
        # Navigate
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(3)

        # Handle consent
        try:
            consent_selectors = ['button:has-text("Consent")', '.fc-button.fc-cta-consent']
            for selector in consent_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        await asyncio.sleep(2)
                        break
                except:
                    continue
        except:
            pass

        # Select English (South Africa)
        locale_select = page.locator('select').first
        await locale_select.select_option(label="English (South Africa) [en-ZA]")
        await asyncio.sleep(5)

        # Enter text
        textarea = page.locator('textarea').first
        await textarea.clear()
        await textarea.fill(test_text)
        print(f"✓ Entered {len(test_text)} characters")
        await asyncio.sleep(1)

        # Click Play
        play_btn = page.locator('#quick-play')
        await play_btn.click()
        print(f"✓ Clicked Play button")

        # Wait a bit
        await asyncio.sleep(15)
        print(f"✓ Waited 15 seconds")

        # Check if download button becomes enabled
        print(f"  Checking if download button enables...")
        try:
            await page.wait_for_function(
                "document.querySelector('#download') && !document.querySelector('#download').disabled",
                timeout=60000  # 60 seconds
            )
            print(f"✅ SUCCESS: {text_length} chars works!")
            result = "SUCCESS"
        except Exception as e:
            print(f"❌ FAILED: {text_length} chars timeout")
            print(f"   Error: {e}")
            result = "FAILED"

    except Exception as e:
        print(f"❌ ERROR: {e}")
        result = "ERROR"
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()

    return result

async def main():
    """Binary search for max character limit"""

    # Test different lengths
    test_lengths = [500, 1000, 1500, 2000, 2500, 3000]

    results = {}
    for length in test_lengths:
        result = await test_char_limit(length)
        results[length] = result

        # If we found a failure, we can narrow down
        if result == "FAILED":
            print(f"\n⚠️  Looks like limit is between {length//2} and {length}")
            break

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    for length, result in results.items():
        print(f"{length:5d} chars: {result}")
    print()

if __name__ == "__main__":
    asyncio.run(main())
