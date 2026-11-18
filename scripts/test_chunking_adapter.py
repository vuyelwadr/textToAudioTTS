#!/usr/bin/env python3
"""
Test script for the text chunking adapter
"""

import sys
import os
from pathlib import Path

# Add current directory to path to import the adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from textChunkingAdapter import WebTTSChunker

REPO_ROOT = Path(__file__).resolve().parents[1]
INDABA_DIR = REPO_ROOT / "library" / "credoMutwa" / "indaba"
DEFAULT_INPUT_FILE = INDABA_DIR / "indaba1.1.txt"
DEFAULT_OUTPUT_DIR = INDABA_DIR / "ttsOnline"

def test_chunking():
    """
    Test the chunking adapter with the indaba1.1.txt file
    """
    print("Testing WebTTS Chunking Adapter")
    print("=" * 50)
    
    # Configuration
    input_file = str(DEFAULT_INPUT_FILE)
    output_dir = str(DEFAULT_OUTPUT_DIR)
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False
    
    try:
        # Initialize chunker
        chunker = WebTTSChunker(
            max_chunk_size=6000,
            min_chunk_size=500,
            output_dir=output_dir
        )
        
        # Process the file
        manifest_path, plan_path = chunker.process_file(
            input_file=input_file,
            prefix="indaba1.1",
            num_workers=6
        )
        
        print("\n✅ Chunking completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Manifest: {manifest_path}")
        print(f"📝 Processing plan: {plan_path}")
        
        # Verify output files were created
        if os.path.exists(manifest_path) and os.path.exists(plan_path):
            print("✅ All output files created successfully")
            return True
        else:
            print("❌ Some output files missing")
            return False
            
    except Exception as e:
        print(f"❌ Error during chunking: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunking()
    sys.exit(0 if success else 1)
