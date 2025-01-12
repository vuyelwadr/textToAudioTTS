import os
import time
from TTS.api import TTS
from pydub import AudioSegment

# Function to split text into chunks
def split_text_into_chunks(text, max_length=250):
    import re
    words = text.split()  # Split text into words
    chunks = []
    current_chunk = ""

    for word in words:
        # Check if adding the word exceeds the max length
        if len(current_chunk) + len(word) + 1 <= max_length:  # +1 accounts for space
            current_chunk += word + " "
        else:
            if current_chunk:  # If the current chunk has content, save it
                chunks.append(current_chunk.strip())
            current_chunk = word + " "  # Start a new chunk with the current word

    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())

    return chunks


# Record the start time
start_time = time.time()

# Initialize TTS with the desired model
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=False
)

# Load text from the file
with open('chapter4.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Split text into chunks
text_chunks = split_text_into_chunks(text, max_length=250)
print(f"There are {len(text_chunks)} chunks")

# Create output directory
output_dir = 'chapter4'
os.makedirs(output_dir, exist_ok=True)

# Generate audio for each chunk and store file paths
audio_files = []
for i, chunk in enumerate(text_chunks):
    output_path = os.path.join(output_dir, f"chapter4_part{i + 1}.wav")
    tts.tts_to_file(
        text=chunk,
        file_path=output_path,
        speaker="Tammie Ema",
        language="en"
    )
    audio_files.append(output_path)

# Combine all the chunked audio files into one
combined = AudioSegment.empty()
for file_path in audio_files:
    audio = AudioSegment.from_wav(file_path)
    combined += audio

# Export the combined audio to a single file
combined_output_path = os.path.join(output_dir, "chapter4.wav")
combined.export(combined_output_path, format="wav")

print(f"All chunks have been combined into '{combined_output_path}'.")

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"There are {len(text_chunks)} chunks")


# All chunks have been combined into 'output_audio/introduction.wav'.
# ch4 Total execution time: 3839.71 seconds There are 276 chunks