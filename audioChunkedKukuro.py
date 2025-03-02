from kokoro import KPipeline
from kokoro.model import KModel # Import KModel for type hinting, might not be strictly necessary
from misaki import en, espeak # Import en and espeak modules
import soundfile as sf
import os
import numpy as np
import multiprocessing
from typing import List, Generator, Tuple, Union, Optional
import torch
import re
from numbers import Number
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger


OUTPUT_DIR = 'civilization/output/1'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_NAME = 'civilization/1.txt'

def custom_chunk_text(text: str, lang_code: str = 'a') -> List[Tuple[str, str]]:
    """
    Chunks text using logic similar to KPipeline's en_tokenize for English.

    Args:
        text (str): The input text to chunk.
        lang_code (str): Language code ('a' or 'b' for English).

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains (text_chunk, phoneme_chunk).
    """
    pipeline = KPipeline(lang_code=lang_code, model=False) # Initialize a "quiet" pipeline for G2P only
    pipeline.g2p.nlp.max_length = 5000000

    if lang_code in 'ab': # English chunking
        _, tokens = pipeline.g2p(text) # Get tokens using kokoro's G2P

        def en_tokenize_like(tokens: List[en.MToken]) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
            tks = []
            pcount = 0
            for t in tokens:
                # American English: ɾ => T (as in KPipeline.en_tokenize)
                t.phonemes = '' if t.phonemes is None else t.phonemes.replace('ɾ', 'T')
                next_ps = t.phonemes + (' ' if t.whitespace else '')
                next_pcount = pcount + len(next_ps.rstrip())
                if next_pcount > 410: # Chunking limit
                    z = KPipeline.waterfall_last(tks, next_pcount)
                    text_chunk = KPipeline.tokens_to_text(tks[:z])
                    ps_chunk = KPipeline.tokens_to_ps(tks[:z])
                    yield text_chunk, ps_chunk, tks[:z]
                    tks = tks[z:]
                    pcount = len(KPipeline.tokens_to_ps(tks))
                    if not tks:
                        next_ps = next_ps.lstrip()
                tks.append(t)
                pcount += len(next_ps)
            if tks:
                text_chunk = KPipeline.tokens_to_text(tks)
                ps_chunk = KPipeline.tokens_to_ps(tks)
                yield text_chunk, ps_chunk, tks

        chunks = []
        for gs, ps, _ in en_tokenize_like(tokens): # Use the en_tokenize_like function
            chunks.append((gs, ps))
        return chunks
    else: # For non-English languages, basic splitting (you can adapt as needed)
        # For simplicity, let's split by newlines if present, otherwise, treat as one chunk.
        if '\n' in text:
            return [(chunk.strip(), pipeline.g2p(chunk.strip())) for chunk in text.split('\n') if chunk.strip()]
        else:
            return [(text, pipeline.g2p(text))]


def generate_audio_segment(segment_data, voice, segment_index, lang_code):
    """
    Generates and saves an audio segment for a given text and phoneme chunk.

    Args:
        segment_data (tuple): (text_chunk, phoneme_chunk) tuple.
        voice (str): The voice to use.
        segment_index (int): Index for file naming.
        lang_code (str): Language code for pipeline.
    """
    text_chunk, phoneme_chunk = segment_data
    pipeline = KPipeline(lang_code=lang_code) # Create pipeline within each process
    generator = pipeline.generate_from_tokens( # Use generate_from_tokens for phoneme input
        phoneme_chunk, voice=voice, speed=1
    )
    audio_segments_filepaths = []
    for i, result in enumerate(generator): # Should be one result per chunk if chunking is correct
        filepath = os.path.join(OUTPUT_DIR, f'segment_{segment_index}_{i}.wav') # Add segment index
        sf.write(filepath, result.audio.cpu().numpy(), 24000) # Save audio from result
        audio_segments_filepaths.append(filepath)
        print(f"Process {segment_index}: Segment {i} saved to {filepath}")
    return audio_segments_filepaths # Return list of filepaths generated in this process


if __name__ == "__main__":
    voice_to_use = 'af_heart'
    lang_code_to_use = 'a'
    num_processes = multiprocessing.cpu_count()

    with open(FILE_NAME, 'r', encoding='utf-8') as file:
        text = file.read()

    text_phoneme_chunks = custom_chunk_text(text, lang_code_to_use)
    total_chunks = len(text_phoneme_chunks)  # Get the total number of chunks
    print(f"Total chunks to process: {total_chunks}")


    segment_tasks = []
    segment_index = 0
    report_interval = 10  # Report progress every 10 chunks (or whatever you set)
    chunks_processed = 0

    for chunk in text_phoneme_chunks:
        segment_tasks.append((chunk, voice_to_use, segment_index, lang_code_to_use))
        segment_index += 1
        chunks_processed += 1

        if chunks_processed % report_interval == 0 or chunks_processed == total_chunks:
            print(f"Chunks processed: {chunks_processed} / {total_chunks}")

    filepaths_lists = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        filepaths_lists = pool.starmap(generate_audio_segment, segment_tasks)

    print("All segments generated and saved in parallel.")

    # --- Joining Logic (Adjusted for new segment naming) ---
    def join_audio_segments_custom_chunk(input_dir=OUTPUT_DIR, output_file='combined_audio_custom_chunk.wav'):
        """Joins WAV audio files from the parallel output directory."""
        audio_segments = []
        sample_rate = None
        segment_index = 0

        while True:
            segment_audio_parts = [] # Collect segments for each main segment index
            segment_part_index = 0
            while True: # Inner loop for segment parts (if a chunk was further segmented)
                filepath = os.path.join(input_dir, f'segment_{segment_index}_{segment_part_index}.wav')
                if not os.path.exists(filepath):
                    if segment_part_index == 0 and not segment_audio_parts: # No parts for this segment index at all
                        break # Move to next segment index
                    else:
                        break # No more parts for this segment index

                try:
                    audio, sr = sf.read(filepath)
                    segment_audio_parts.append(audio)
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        raise ValueError(f"Sample rate mismatch: {filepath}")
                except sf.LibsndfileError as e:
                    print(f"Error reading {filepath}: {e}")
                segment_part_index += 1
            if not segment_audio_parts and segment_index > len(text_phoneme_chunks): # Stop if no segments for current and subsequent indices
                break # No more segments
            combined_segment_audio = np.concatenate(segment_audio_parts, axis=0) if segment_audio_parts else np.array([]) # Join parts of segment
            audio_segments.append(combined_segment_audio) # Add joined segment audio to list
            segment_index += 1


        if not audio_segments:
            print(f"No audio segments found in '{input_dir}' to join.")
            return

        combined_audio = np.concatenate(audio_segments, axis=0)

        try:
            sf.write(f'{input_dir}/{output_file}', combined_audio, sample_rate)
            print(f"Successfully joined audio segments into '{output_file}'")
        except sf.LibsndfileError as e:
            print(f"Error writing combined audio: {e}")

    join_audio_segments_custom_chunk()
    
# https://github.com/remsky/Kokoro-FastAPI

# Total execution time: 170.87 seconds
# There are 154 chunks

# All chunks processed.
# All chunks have been combined into 'jordan_kokoro/1.wav'.
# Total execution time: 6953.20 seconds
# There are 4894 chunks