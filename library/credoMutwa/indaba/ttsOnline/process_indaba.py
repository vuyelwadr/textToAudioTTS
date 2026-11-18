#!/usr/bin/env python3
"""
Example script to process the Credo Mutwa Indaba text using TTS Online automation
"""

import os
import sys
from pathlib import Path

def main():
    """Process the indaba1.1.txt file with TTS Online automation"""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Input file path (indaba1.1.txt is in the parent directory)
    input_file = script_dir.parent / "indaba1.1.txt"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please ensure indaba1.1.txt exists in the indaba directory")
        sys.exit(1)
    
    print("🎤 Credo Mutwa Indaba TTS Processing")
    print("="*50)
    print(f"Input file: {input_file}")
    print(f"File size: {input_file.stat().st_size:,} bytes")
    
    # Import the automation script
    try:
        sys.path.insert(0, str(script_dir))
        from ttsOnline_automation import TTSOnlineAutomator
    except ImportError as e:
        print(f"❌ Failed to import automation script: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create output directory specific to indaba
    output_dir = script_dir / "indaba_tts_output"
    
    # Create automator
    automator = TTSOnlineAutomator(
        output_dir=str(output_dir),
        headless=True  # Set to False for debugging
    )
    
    print(f"Output directory: {output_dir}")
    print("Starting processing...")
    print()
    
    try:
        # Process the entire file
        result = automator.process_text_file(
            str(input_file),
            start_chunk=0,
            end_chunk=None  # Process all chunks
        )
        
        # Print results
        print()
        print("="*50)
        print("✅ PROCESSING COMPLETE")
        print("="*50)
        print(f"Total chunks processed: {result['total_chunks']}")
        print(f"Successfully generated: {len(result['successful_files'])} files")
        print(f"Failed chunks: {len(result['failed_chunks'])}")
        print(f"Success rate: {result['success_rate']:.1f}%")
        print()
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Session report: Check for session_report_*.json")
        print(f"📝 Detailed log: Check for tts_online_log_*.txt")
        
        if result['failed_chunks']:
            print()
            print(f"⚠️  Failed chunks: {result['failed_chunks']}")
            print("To retry failed chunks:")
            for chunk_num in result['failed_chunks']:
                print(f"  python ttsOnline_automation.py ../indaba1.1.txt --start-chunk {chunk_num} --end-chunk {chunk_num + 1}")
        
        print()
        print("Next steps:")
        print("1. Listen to the generated audio files")
        print("2. If needed, retry failed chunks")
        print("3. Join audio files using the project's audio joining scripts")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        print("You can resume by running:")
        print(f"  python ttsOnline_automation.py ../indaba1.1.txt --start-chunk <last_successful_chunk>")
        
    except Exception as e:
        print(f"\n❌ Fatal error during processing: {e}")
        print("Please check the log files for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
