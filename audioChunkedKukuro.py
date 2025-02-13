import os
import time
from pydub import AudioSegment
import multiprocessing
import soundfile as sf
from kokoro import KPipeline  # Import Kokoro
from tqdm import tqdm # Import tqdm for progress bar


INPUT_FILE = 'jordan1.txt'
OUTPUT_FILE = '1.wav'
OUTPUT_DIR = 'jordan_kokoro_2x'

# Initialize Kokoro pipeline (moved outside process_chunk for efficiency)
pipeline = KPipeline(lang_code='a')  # American English

def process_chunk(chunk_index, chunk, output_dir):
    """
    Processes a single text chunk using Kokoro TTS and generates an audio file.

    Args:
        chunk_index (int): Index of the chunk.
        chunk (str): Text chunk to process.
        output_dir (str): Directory to save the audio file.

    Returns:
        str: Path to the generated audio file.
    """
    # Generate audio using Kokoro pipeline
    generator = pipeline(
        chunk, voice='af_heart',  # You can change voice here, 'af_heart' is an example
        speed=2, split_pattern=r'\n+'  # keep split_pattern in case chunks contain newlines
    )
    audio_chunk = None  # Initialize audio_chunk to None

    for i, (gs, ps, audio) in enumerate(generator):
        audio_chunk = audio  # Get the audio from the generator (assuming only one chunk per text chunk)
        break  # Exit loop after getting the audio once per chunk

    if audio_chunk is not None:
        output_path = os.path.join(output_dir, f"chapter4_part{chunk_index + 1}.wav")  # Keep chapter4 naming for consistency, you can change it
        sf.write(output_path, audio_chunk, 24000)  # Save audio using soundfile, 24000 rate from Kokoro example
        return output_path
    else:
        print(f"Warning: No audio generated for chunk {chunk_index + 1}. Check the chunk content.")
        return None  # Return None if no audio was generated


# Function to split text into chunks (same as before)
def split_text_into_chunks(text, max_length=250):
    import re
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += word + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = word + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Load text from the file
    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text into chunks
    text_chunks = split_text_into_chunks(text, max_length=250)
    num_chunks = len(text_chunks) # Get total number of chunks
    print(f"There are {num_chunks} chunks")

    # Create output directory
    output_dir = OUTPUT_DIR  # Changed output directory name to avoid confusion
    os.makedirs(output_dir, exist_ok=True)

    # --- Multiprocessing Implementation with Progress Tracking ---
    audio_files = []
    with multiprocessing.Pool() as pool:
        processes = []
        for i, chunk in enumerate(text_chunks):
            process = pool.apply_async(process_chunk, args=(i, chunk, output_dir))
            processes.append(process)

        print("Waiting for all chunks to be processed...")
        processed_chunks_count = 0
        # Use tqdm to create a progress bar
        for process in tqdm(processes, total=num_chunks, desc="Processing Chunks"):
            audio_file_path = process.get()
            if audio_file_path:  # Only append if audio file path is not None (audio was generated)
                audio_files.append(audio_file_path)
            processed_chunks_count += 1
            # print(f"Processed chunks: {processed_chunks_count}/{num_chunks}") # Basic progress print - replaced by tqdm
        print("All chunks processed.")
    # --- End Multiprocessing Implementation ---

    # Combine all the chunked audio files into one
    combined = AudioSegment.empty()
    for file_path in audio_files:
        audio = AudioSegment.from_wav(file_path)
        combined += audio

    # Export the combined audio to a single file
    combined_output_path = os.path.join(output_dir, OUTPUT_FILE)  # Keep 01.wav output name
    combined.export(combined_output_path, format="wav")

    print(f"All chunks have been combined into '{combined_output_path}'.")

    # Record the end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    print(f"There are {len(text_chunks)} chunks")

# https://github.com/remsky/Kokoro-FastAPI

# Total execution time: 170.87 seconds
# There are 154 chunks

# All chunks processed.
# All chunks have been combined into 'jordan_kokoro/1.wav'.
# Total execution time: 6953.20 seconds
# There are 4894 chunks