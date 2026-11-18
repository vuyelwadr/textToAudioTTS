
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import time
import torch
from audioChunkedKukuro import process_file

# Create a dummy input file
TEST_FILE = "test_gen_input.txt"
with open(TEST_FILE, "w") as f:
    f.write("This is a test of the Kokoro TTS system running on Apple Silicon. Metal Performance Shaders should be active.")

def test_generation():
    print("--- Starting Generation Test ---")
    if torch.backends.mps.is_available():
        print("MPS is available and should be used.")
    else:
        print("MPS is NOT available. This test might be slow or fail if logic is wrong.")

    start_time = time.time()
    try:
        # Run the processing
        output_files = process_file(TEST_FILE, voice='af_alloy')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n--- Test Completed in {duration:.2f} seconds ---")
        print(f"Generated {len(output_files)} segments.")
        for f in output_files:
            print(f" - {f}")
            
    except Exception as e:
        print(f"!!! Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
