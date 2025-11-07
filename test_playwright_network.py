#!/usr/bin/env python3
"""Test different Playwright network configurations"""

import asyncio
from playwright.async_api import async_playwright

async def test_config(name, args):
    """Test a specific browser configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Args: {args}")
    print(f"{'='*70}")

    playwright = None
    browser = None

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=args
        )
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        page.set_default_timeout(15000)

        print("🌐 Attempting to navigate...")
        await page.goto("https://www.text-to-speech.online/", wait_until="domcontentloaded")

        title = await page.title()
        print(f"✅ SUCCESS! Page title: {title}")

        await context.close()
        await browser.close()
        await playwright.stop()

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        if browser:
            await browser.close()
        if playwright:
            await playwright.stop()
        return False

async def main():
    print("Testing different Playwright network configurations...")

    configs = [
        ("Default (no special args)", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled"
        ]),

        ("Direct proxy bypass", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--proxy-server=direct://"
        ]),

        ("No proxy server", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--no-proxy-server"
        ]),

        ("Disable network service", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=NetworkService"
        ]),

        ("Ignore certificate errors", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--ignore-certificate-errors"
        ]),

        ("All network bypass flags", [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--proxy-server=direct://",
            "--no-proxy-server",
            "--ignore-certificate-errors",
            "--disable-web-security"
        ])
    ]

    results = []
    for name, args in configs:
        success = await test_config(name, args)
        results.append((name, success))
        await asyncio.sleep(2)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for name, success in results:
        status = "✅ WORKS" if success else "❌ FAILED"
        print(f"{status}: {name}")

    # Find working config
    working = [name for name, success in results if success]
    if working:
        print(f"\n🎉 Working configuration found: {working[0]}")
    else:
        print("\n❌ No working configuration found")

if __name__ == "__main__":
    asyncio.run(main())
