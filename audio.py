# 1️⃣ Install kokoro
# 2️⃣ Install espeak, used for English OOD fallback and some non-English languages
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇧🇷 'p' => Brazilian Portuguese pt-br

# 3️⃣ Initalize a pipeline
from kokoro import KPipeline
import soundfile as sf
import os
import numpy as np
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice

# This text is for demonstration purposes only, unseen during training

with open('jordan1.txt', 'r', encoding='utf-8') as file:
    text = file.read()


# 4️⃣ Generate, display, and save audio files in a loop.
generator = pipeline(
    text, voice='af_heart', # <= change voice here
    speed=1, split_pattern=''
)
for i, (gs, ps, audio) in enumerate(generator):
    print(i)  # i => index
    print(gs) # gs => graphemes/text
    print(ps) # ps => phonemes
    sf.write(f'out/{i}.wav', audio, 24000) # save each audio file


def join_audio_segments(input_dir='out', output_file='combined_audio.wav'):
    """
    Joins WAV audio files from a directory into a single seamless audio file.

    Args:
        input_dir (str): Directory containing the segmented WAV files (e.g., 'out').
        output_file (str): Path to save the combined audio file (e.g., 'combined_audio.wav').
    """
    audio_segments = []
    sample_rate = None  # To store the sample rate from the first segment

    file_index = 0
    while True:
        filepath = os.path.join(input_dir, f'{file_index}.wav')
        if not os.path.exists(filepath):
            break  # Stop when no more numbered files are found

        try:
            audio, sr = sf.read(filepath)
            audio_segments.append(audio)
            if sample_rate is None:
                sample_rate = sr  # Set sample rate from the first file
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch between files. File: {filepath}, Expected SR: {sample_rate}, Found SR: {sr}")
        except sf.LibsndfileError as e:
            print(f"Error reading file {filepath}: {e}")
            # You might want to handle errors differently, e.g., skip the file or exit.
            # For now, we'll just print the error and continue.
        file_index += 1

    if not audio_segments:
        print(f"No audio segments found in '{input_dir}' to join.")
        return

    # Concatenate all audio segments into a single NumPy array
    combined_audio = np.concatenate(audio_segments, axis=0)

    # Write the combined audio to a new WAV file
    try:
        sf.write(f'{input_dir}/{output_file}', combined_audio, sample_rate)
        print(f"Successfully joined {file_index} audio segments into '{output_file}'")
    except sf.LibsndfileError as e:
        print(f"Error writing combined audio to '{output_file}': {e}")
join_audio_segments()