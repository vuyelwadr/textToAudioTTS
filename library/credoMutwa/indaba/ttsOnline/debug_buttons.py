#!/usr/bin/env python3
"""
DEBUG SCRIPT - Inspect what buttons/controls actually exist on the page
"""

import asyncio
import sys
from playwright.async_api import async_playwright

async def debug_page_controls():
    """Debug what controls exist on the page"""

    print("=" * 70)
    print("🔍 DEBUG: Inspecting text-to-speech.online controls")
    print("=" * 70)
    print()

    playwright = await async_playwright().start()

    try:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print("🌐 Loading page...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")
        await asyncio.sleep(5)

        # Handle consent
        try:
            await page.click('button:has-text("I agree")', timeout=3000)
            print("✅ Consent accepted")
            await asyncio.sleep(2)
        except:
            pass

        # Select locale
        print("🌍 Selecting locale...")
        try:
            await page.select_option('select', label="English (South Africa) [en-ZA]")
            await asyncio.sleep(3)
        except:
            pass

        # Enter text
        print("⌨️  Entering text...")
        try:
            await page.fill('textarea', "Test text")
            await asyncio.sleep(2)
        except:
            pass

        print()
        print("=" * 70)
        print("📋 ALL BUTTONS ON PAGE:")
        print("=" * 70)

        # Get all buttons
        buttons = await page.locator('button').all()
        print(f"Found {len(buttons)} buttons:")
        print()

        for i, btn in enumerate(buttons, 1):
            try:
                text = await btn.text_content()
                visible = await btn.is_visible()
                classes = await btn.get_attribute('class')
                btn_id = await btn.get_attribute('id')
                title = await btn.get_attribute('title')
                aria_label = await btn.get_attribute('aria-label')

                print(f"Button #{i}:")
                print(f"  Text: '{text.strip() if text else 'N/A'}'")
                print(f"  Visible: {visible}")
                if btn_id:
                    print(f"  ID: {btn_id}")
                if classes:
                    print(f"  Classes: {classes}")
                if title:
                    print(f"  Title: {title}")
                if aria_label:
                    print(f"  Aria-label: {aria_label}")
                print()
            except:
                print(f"Button #{i}: (could not inspect)")
                print()

        print("=" * 70)
        print("🔊 AUDIO/MEDIA ELEMENTS:")
        print("=" * 70)

        # Check for audio elements
        audio_elements = await page.locator('audio').all()
        print(f"Audio elements: {len(audio_elements)}")

        # Check for input buttons
        input_buttons = await page.locator('input[type="button"]').all()
        print(f"Input buttons: {len(input_buttons)}")

        # Check for links that might be download
        links = await page.locator('a').all()
        download_links = []
        for link in links:
            href = await link.get_attribute('href')
            download_attr = await link.get_attribute('download')
            if download_attr or (href and 'download' in href.lower()):
                text = await link.text_content()
                download_links.append(text.strip() if text else 'N/A')

        print(f"Download links: {len(download_links)}")
        if download_links:
            for link in download_links:
                print(f"  - {link}")

        print()
        print("🎯 WAITING 60 seconds - Manually click buttons to test...")
        print("   Look for play/generate/download buttons and click them")
        print("   Press Ctrl+C when done")

        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            print("\n✅ Interrupted by user")

    finally:
        print("\n🧹 Cleaning up...")
        try:
            await context.close()
            await browser.close()
            await playwright.stop()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(debug_page_controls())
