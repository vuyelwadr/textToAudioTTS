#!/usr/bin/env python3
"""
Simple working TTS downloader using requests - fallback method
This demonstrates a working download mechanism
"""

import os
import sys
import time
import requests
from pathlib import Path


def download_with_requests(url, output_path):
    """
    Simple download using requests library
    This is a fallback method that works reliably
    """
    print(f"⬇️  Downloading from: {url}")
    print(f"💾 Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}%", end='', flush=True)

        print()  # New line after progress

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"✅ SUCCESS! Downloaded {file_size:,} bytes")
                return True

        return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """
    Demo: Download a sample audio file to show the mechanism works
    """
    print("=" * 70)
    print("🎙️  Simple TTS Downloader - Demo Version")
    print("=" * 70)

    # Create output directory
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Demo: Download a small sample audio file
    # This demonstrates that our download mechanism works
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"demo_audio_{timestamp}.mp3")

    # Using a sample audio URL (replace with actual TTS API URL)
    demo_url = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"

    print("\n📝 This is a demo showing the download mechanism works")
    print(f"🔗 Test URL: {demo_url}\n")

    success = download_with_requests(demo_url, output_path)

    print("=" * 70)
    if success:
        print("🎉 DOWNLOAD SUCCESSFUL!")
        print(f"📁 File saved to: {output_path}")
        print("\n💡 The download mechanism works!")
        print("   Next step: Integrate with TTS service API")
        print("=" * 70)
        return 0
    else:
        print("❌ DOWNLOAD FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
