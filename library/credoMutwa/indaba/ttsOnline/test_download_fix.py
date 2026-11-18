#!/usr/bin/env python3
"""
Test script to verify the Chrome download fix
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path so we can import the working module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from working_tts_automation import WorkingTTSAutomator

def test_download_fix():
    """Test the fixed download mechanism"""
    print("🧪 Testing Chrome download fix...")
    
    # Create automator with visible browser for debugging
    download_dir = "./tts_output_test"
    automator = WorkingTTSAutomator(
        download_dir=download_dir, 
        headless=False  # Use visible browser for testing
    )
    
    try:
        test_text = "Hello, this is a test of the fixed Chrome download mechanism. We are using English South Africa voice to verify that files are now downloaded to the correct directory."
        
        print(f"📁 Configured download directory: {Path(download_dir).absolute()}")
        print(f"📝 Test text length: {len(test_text)} characters")
        print("🚀 Starting TTS automation test...")
        
        result = automator.process_chunk(
            text=test_text,
            chunk_index=1,
            locale="English (South Africa)",
            voice_gender="male"
        )
        
        if result and os.path.exists(result):
            file_size = os.path.getsize(result)
            file_path = Path(result)
            
            print("✅ SUCCESS: Chrome download fix working!")
            print(f"   📄 File: {file_path.name}")
            print(f"   📍 Location: {file_path.absolute()}")
            print(f"   📊 Size: {file_size:,} bytes")
            print(f"   🕐 Age: {time.time() - file_path.stat().st_mtime:.1f} seconds")
            
            # Verify file is in the expected directory
            if str(file_path.parent).endswith(download_dir.lstrip("./")):
                print("   ✅ Download directory: CORRECT")
            else:
                print(f"   ⚠️  Download directory: UNEXPECTED (wanted: {download_dir})")
                
            return True
        else:
            print("❌ FAILED: No audio file was downloaded")
            
            # Check if any files were created anywhere
            print("\n🔍 Searching for any recent audio files...")
            import glob
            from datetime import datetime, timedelta
            
            # Common download locations
            search_dirs = [
                download_dir,
                str(Path.home() / "Downloads"),
                str(Path.home() / "Desktop"),
                "/tmp"
            ]
            
            cutoff_time = datetime.now() - timedelta(minutes=5)
            found_files = False
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for ext in ['*.mp3', '*.wav', '*.ogg', '*.m4a']:
                        pattern = os.path.join(search_dir, ext)
                        for file_path in glob.glob(pattern):
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mtime > cutoff_time:
                                file_size = os.path.getsize(file_path)
                                print(f"   📁 Found: {file_path} ({file_size:,} bytes, {file_mtime})")
                                found_files = True
            
            if not found_files:
                print("   📁 No recent audio files found in common locations")
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        print("🛑 Cleaning up...")
        automator.end_session()
        
        # Wait a moment for cleanup
        time.sleep(2)

if __name__ == "__main__":
    success = test_download_fix()
    if success:
        print("\n🎉 Chrome download fix verification PASSED!")
    else:
        print("\n💥 Chrome download fix verification FAILED!")
    
    sys.exit(0 if success else 1)
