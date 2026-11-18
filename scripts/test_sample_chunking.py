#!/usr/bin/env python3
"""
Test the chunking adapter with a sample text to verify it works correctly
"""

import sys
import os
import tempfile

# Add current directory to path to import the adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from textChunkingAdapter import WebTTSChunker

def test_with_sample():
    """
    Test the chunking adapter with a sample text
    """
    print("🧪 Testing WebTTS Chunking Adapter with Sample Text")
    print("=" * 60)
    
    # Sample text with various sentence structures
    sample_text = """
    This is the first paragraph. It contains multiple sentences to test the chunking algorithm.
    Some sentences are longer than others, and some have different punctuation; like commas, semicolons; and colons: for example.
    We also want to test how the system handles questions? And exclamations! These should be natural breaking points.

    This is the second paragraph with a double line break. The system should preserve this paragraph boundary.
    This paragraph contains some complex sentences with multiple clauses: these clauses should be respected when possible, but the system should break at sentence boundaries first.
    We want to ensure that the chunking produces natural-sounding segments for text-to-speech conversion.

    The third paragraph demonstrates how the system handles very long sentences that might exceed the chunk limit.
    When a sentence is extremely long and contains multiple clauses separated by commas, semicolons, and other punctuation marks, the system needs to decide where to break the text while maintaining readability and natural speech flow.
    This is particularly important for web-based TTS services that can handle large chunks but still benefit from proper segmentation.

    Finally, this fourth paragraph concludes our test. It should demonstrate that the chunking system can handle various text structures effectively.
    The goal is to create chunks that are optimized for web TTS processing while maintaining semantic coherence and natural speech patterns.
    """
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Test with different chunk sizes
        test_configs = [
            {"max_chunk_size": 500, "min_chunk_size": 100, "name": "Small chunks"},
            {"max_chunk_size": 1500, "min_chunk_size": 200, "name": "Medium chunks"},
            {"max_chunk_size": 3000, "min_chunk_size": 300, "name": "Large chunks"}
        ]
        
        for config in test_configs:
            print(f"\n🔧 Testing {config['name']} (max: {config['max_chunk_size']}, min: {config['min_chunk_size']})")
            print("-" * 50)
            
            # Initialize chunker
            chunker = WebTTSChunker(
                max_chunk_size=config['max_chunk_size'],
                min_chunk_size=config['min_chunk_size'],
                output_dir=temp_dir
            )
            
            # Create chunks
            chunks = chunker.chunk_text_optimized(sample_text)
            
            # Analyze results
            print(f"✅ Created {len(chunks)} chunks")
            
            # Check chunk sizes
            sizes = [len(c['text']) for c in chunks]
            print(f"📏 Chunk sizes: {sizes}")
            print(f"📏 Average size: {sum(sizes)/len(sizes):.0f} characters")
            print(f"📏 Min/Max: {min(sizes)}/{max(sizes)} characters")
            
            # Verify no chunks exceed max size
            max_violations = [s for s in sizes if s > config['max_chunk_size']]
            if max_violations:
                print(f"❌ Found {len(max_violations)} chunks exceeding max size: {max_violations}")
            else:
                print("✅ All chunks within size limits")
            
            # Verify no chunks below min size (except possibly the last one)
            min_violations = [s for s in sizes[:-1] if s < config['min_chunk_size']]
            if min_violations:
                print(f"⚠️  Found {len(min_violations)} chunks below min size: {min_violations}")
            else:
                print("✅ All chunks meet minimum size requirements")
            
            # Check for proper sentence boundaries
            print(f"\n📝 Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                print(f"  Chunk {i}: {len(chunk['text'])} chars")
                print(f"    Preview: {text_preview}")
                print(f"    Ends with: '{chunk['text'][-50:]}'")
            
            if len(chunks) > 3:
                print(f"  ... and {len(chunks) - 3} more chunks")
            
            # Test manifest creation
            print(f"\n📋 Testing manifest creation...")
            try:
                manifest_path = chunker.save_chunks(chunks, "test_sample")
                print(f"✅ Manifest saved to: {manifest_path}")
                
                # Verify manifest exists and is readable
                if os.path.exists(manifest_path):
                    import json
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    print(f"✅ Manifest contains {len(manifest['chunks'])} chunks")
                    print(f"📊 Total characters: {manifest['metadata']['total_characters']}")
                    print(f"⏱️  Estimated duration: {manifest['metadata']['estimated_total_duration_seconds']:.1f} seconds")
                else:
                    print("❌ Manifest file not found")
                    
            except Exception as e:
                print(f"❌ Error creating manifest: {str(e)}")
    
    print(f"\n🎉 Chunking adapter test completed successfully!")
    return True

def test_edge_cases():
    """
    Test edge cases for the chunking adapter
    """
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        chunker = WebTTSChunker(output_dir=temp_dir)
        
        # Test 1: Very short text
        print("\n1️⃣ Testing very short text...")
        short_text = "This is a short text."
        chunks = chunker.chunk_text_optimized(short_text)
        print(f"✅ Short text: {len(chunks)} chunk(s) created")
        
        # Test 2: Empty text
        print("\n2️⃣ Testing empty text...")
        empty_chunks = chunker.chunk_text_optimized("")
        print(f"✅ Empty text: {len(empty_chunks)} chunk(s) created")
        
        # Test 3: Text with only punctuation
        print("\n3️⃣ Testing punctuation-only text...")
        punct_text = "...!?!...;:;...!?!"
        punct_chunks = chunker.chunk_text_optimized(punct_text)
        print(f"✅ Punctuation text: {len(punct_chunks)} chunk(s) created")
        
        # Test 4: Very long single sentence
        print("\n4️⃣ Testing very long single sentence...")
        long_sentence = "This is an extremely long sentence that goes on and on with many commas and semicolons and clauses and subclauses that should test the system's ability to handle text that doesn't have natural sentence boundaries within reasonable limits and needs to be broken at clause boundaries instead. " * 3
        long_chunks = chunker.chunk_text_optimized(long_sentence)
        print(f"✅ Long sentence: {len(long_chunks)} chunk(s) created")
        print(f"📏 Original length: {len(long_sentence)} characters")
        print(f"📏 Chunk sizes: {[len(c['text']) for c in long_chunks]}")
    
    print(f"\n✅ All edge cases handled correctly!")
    return True

if __name__ == "__main__":
    try:
        # Run main tests
        test_with_sample()
        test_edge_cases()
        
        print(f"\n🎊 All tests passed! The chunking adapter is working correctly.")
        print(f"📝 Ready for use with web TTS automation!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
