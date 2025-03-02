import os
import numpy as np
import multiprocessing
import warnings
from typing import List, Generator, Tuple, Union, Optional
import torch
import re
import time
import soundfile as sf
from tqdm import tqdm
from functools import lru_cache
from kokoro import KPipeline
from kokoro.model import KModel
from misaki import en, espeak
import gc  # For garbage collection
import sys
import threading

# Filter out specific PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                        module="torch.nn.utils.weight_norm")

# Constants - Moved to top for easier configuration
OUTPUT_DIR = 'civilization/output/2'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_NAME = 'civilization/2.txt'
BATCH_SIZE = 10  # Process chunks in batches for better performance
MAX_THREADS_PER_PROCESS = 2  # Limit threads per process to avoid memory issues
VOICE_TO_USE = 'af_heart'
LANG_CODE = 'a'  # 'a' for English
SAMPLE_RATE = 24020
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU

# Configure pipeline settings for caching
cached_pipelines = {}

def get_pipeline(lang_code, model=True):
    """Get or create a cached pipeline instance to avoid redundant initialization"""
    global cached_pipelines
    key = (lang_code, model)
    if key not in cached_pipelines:
        cached_pipelines[key] = KPipeline(lang_code=lang_code, model=model)
        if model:
            # Optimize for batch processing
            cached_pipelines[key].model.eval()  # Ensure model is in evaluation mode
            if USE_GPU:
                cached_pipelines[key].model.cuda()
    return cached_pipelines[key]

# Improved text chunking with LRU cache for phoneme generation
@lru_cache(maxsize=1024)
def get_phonemes_for_text(text, lang_code):
    """Cached phoneme generation to avoid redundant G2P processing"""
    pipeline = get_pipeline(lang_code, model=False)
    return pipeline.g2p(text)

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

    print(f"Chunking text using language code: {lang_code}")

    if lang_code in 'ab': # English chunking
        print("Processing English text...")
        _, tokens = pipeline.g2p(text) # Get tokens using kokoro's G2P
        print(f"Obtained {len(tokens)} tokens from G2P")

        def en_tokenize_like(tokens: List[en.MToken]) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
            tks = []
            pcount = 0
            for t in tqdm(tokens, desc="Tokenizing text"):
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
        print(f"Created {len(chunks)} text chunks")
        return chunks
    else: # For non-English languages, basic splitting (you can adapt as needed)
        print("Processing non-English text...")
        # For simplicity, let's split by newlines if present, otherwise, treat as one chunk.
        if '\n' in text:
            chunks = []
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in tqdm(lines, desc="Processing text lines"):
                chunks.append((line, pipeline.g2p(line)))
            return chunks
        else:
            return [(text, pipeline.g2p(text))]

# Simplified progress reporting with basic types
def create_progress_tracker(num_processes, total_batches, total_chunks):
    """Create a dict of shared objects that can be safely passed between processes"""
    manager = multiprocessing.Manager()
    return {
        'process_status': manager.dict(),
        'completed_batches': manager.Value('i', 0),
        'stop_flag': manager.Value('b', False),
        'start_time': time.time(),  # Add overall start time
        'batch_times': manager.dict()  # Track time for each batch
    }

def update_progress(tracker, process_id, batch_index, segment_index, total_segments):
    """Update progress info in the shared dict"""
    if tracker and 'process_status' in tracker:
        # Record batch start time if this is the first segment
        if segment_index == 0 and 'batch_times' in tracker:
            batch_key = f"{process_id}_{batch_index}"
            if batch_key not in tracker['batch_times']:
                tracker['batch_times'][batch_key] = time.time()
        
        tracker['process_status'][process_id] = {
            'batch': batch_index,
            'segment': segment_index,
            'total_segments': total_segments,
            'status': 'working',
            'batch_key': f"{process_id}_{batch_index}"  # Store batch key for time lookup
        }

def complete_batch(tracker, process_id):
    """Mark a batch as completed"""
    if tracker and 'completed_batches' in tracker:
        tracker['completed_batches'].value += 1
        if 'process_status' in tracker:
            tracker['process_status'][process_id]['status'] = 'idle'

def generate_audio_batch(process_id, batch_index, batch_data, total_batches, total_chunks, progress_tracker=None):
    """
    Process a batch of segments at once for better performance
    Uses a simplified progress tracker
    """
    start_time = time.time()
    pipeline = get_pipeline(LANG_CODE, model=True)
    
    # Move model to GPU if available
    if USE_GPU and not next(pipeline.model.parameters()).is_cuda:
        pipeline.model.cuda()
    
    filepaths = []
    total_segments = len(batch_data)
    
    # Update progress tracker at start
    update_progress(progress_tracker, process_id, batch_index, 0, total_segments)
    
    # Process each segment in the batch
    for segment_index, (segment_data, voice) in enumerate(batch_data):
        # Update progress
        update_progress(progress_tracker, process_id, batch_index, segment_index, total_segments)
        
        text_chunk, phoneme_chunk = segment_data
        global_segment_index = batch_index * BATCH_SIZE + segment_index
        
        try:
            # Generate audio
            generator = pipeline.generate_from_tokens(
                phoneme_chunk, 
                voice=voice, 
                speed=1
            )
            
            for i, result in enumerate(generator):
                filepath = os.path.join(OUTPUT_DIR, f'segment_{global_segment_index}_{i}.wav')
                # Convert to numpy and save
                if USE_GPU:
                    audio_data = result.audio.cpu().numpy()
                else:
                    audio_data = result.audio.numpy()
                
                sf.write(filepath, audio_data, SAMPLE_RATE)
                filepaths.append(filepath)
        except Exception as e:
            print(f"Error generating audio for segment {global_segment_index}: {str(e)}")
    
    # Clear CUDA cache after batch processing
    if USE_GPU:
        torch.cuda.empty_cache()
    
    # Mark batch as completed
    complete_batch(progress_tracker, process_id)
    
    elapsed = time.time() - start_time
    return filepaths

def join_audio_segments_streaming(input_dir=OUTPUT_DIR, output_file='combined_audio.wav'):
    """
    Join audio files using memory-efficient streaming approach
    """
    print("Joining audio segments...")
    start_time = time.time()
    
    # First, identify all segment files and sort them
    segment_files = []
    for file in os.listdir(input_dir):
        if file.startswith('segment_') and file.endswith('.wav'):
            # Parse segment indices for proper ordering
            parts = file.split('_')
            if len(parts) >= 3:
                segment_index = int(parts[1])
                part_index = int(parts[2].split('.')[0])
                segment_files.append((segment_index, part_index, os.path.join(input_dir, file)))
    
    if not segment_files:
        print(f"No audio segments found in '{input_dir}' to join.")
        return
    
    # Sort by segment index and then by part index
    segment_files.sort()
    
    # Get sample rate from first file
    _, sample_rate = sf.read(segment_files[0][2], frames=1)
    
    # Open output file for streaming writing
    with sf.SoundFile(os.path.join(input_dir, output_file), 
                      'w', 
                      samplerate=sample_rate,
                      channels=1, 
                      format='WAV') as outfile:
        
        # Process files in batches to minimize memory usage
        for i, (_, _, filepath) in enumerate(tqdm(segment_files, desc="Joining audio")):
            # Read audio in chunks and write directly to output
            with sf.SoundFile(filepath) as infile:
                while True:
                    chunk = infile.read(32000)  # Read in chunks of 32000 samples
                    if not len(chunk):
                        break
                    outfile.write(chunk)
    
    elapsed = time.time() - start_time
    print(f"Successfully joined {len(segment_files)} audio segments into '{output_file}' in {elapsed:.2f} seconds")

def process_batch_wrapper(args):
    """
    Wrapper function for multiprocessing.Pool.imap
    Extracts arguments and passes them to generate_audio_batch
    """
    return generate_audio_batch(*args)

# Display function that runs in main process
def display_progress(tracker, num_processes, total_batches, total_chunks):
    """Display progress bars in the main process with time tracking"""
    # Clear screen portion for our progress display
    sys.stdout.write("\033[?25l")  # Hide cursor
    
    try:
        start_time = tracker['start_time']
        
        # Create master progress bar with time info
        master_bar = tqdm(
            total=total_batches,
            position=num_processes,
            desc="Overall progress",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches ({percentage:.1f}%) "
                      "[Elapsed: {elapsed}, ETA: {remaining}]"
        )
        master_bar.update(0)  # Initialize master bar
        
        # Create process bars (one per process)
        process_bars = []
        for i in range(num_processes):
            bar = tqdm(
                total=10,  # Default to batch size
                position=i,
                desc=f"Process {i:2d}",
                bar_format="{desc}: {bar} {n_fmt}/{total_fmt} segs | {postfix}"
            )
            process_bars.append(bar)
            bar.update(0)  # Initialize with 0
        
        last_completed = 0
        batch_times = {}  # Local cache of batch start times
        
        # Update display until stopped
        while not tracker['stop_flag'].value:
            # Update overall time tracking
            elapsed = time.time() - start_time
            completed = tracker['completed_batches'].value
            
            # Calculate overall progress metrics
            if completed > 0:
                avg_time_per_batch = elapsed / completed
                remaining_batches = total_batches - completed
                eta = avg_time_per_batch * remaining_batches
                eta_str = f"{eta:.1f}s"
                if eta > 60:
                    eta_str = f"{eta/60:.1f}m"
                if eta > 3600:
                    eta_str = f"{eta/3600:.1f}h"
            else:
                eta_str = "calculating..."
            
            # Update master progress if needed
            if completed > last_completed:
                master_bar.update(completed - last_completed)
                master_bar.set_postfix_str(f"Elapsed: {elapsed:.1f}s, ETA: {eta_str}")
                last_completed = completed
            
            # Update batch times from tracker
            if 'batch_times' in tracker:
                batch_times.update(tracker['batch_times'])
            
            # Update each process bar
            if 'process_status' in tracker:
                for proc_id, status in tracker['process_status'].items():
                    if proc_id < len(process_bars):
                        bar = process_bars[proc_id]
                        
                        if status.get('batch') is not None:
                            batch_num = status['batch'] + 1
                            total_segments = status.get('total_segments', 10)
                            segment = status.get('segment', 0) + 1
                            
                            # Calculate batch elapsed time
                            batch_key = status.get('batch_key')
                            batch_elapsed = 0
                            batch_eta = "calculating..."
                            
                            if batch_key and batch_key in batch_times:
                                batch_elapsed = time.time() - batch_times[batch_key]
                                
                                # Estimate batch completion time
                                if segment > 1:  # Need at least one segment processed to estimate
                                    time_per_segment = batch_elapsed / segment
                                    segments_left = total_segments - segment
                                    batch_eta = f"{segments_left * time_per_segment:.1f}s"
                            
                            # Update desc and total if needed
                            bar.set_description(f"Process {proc_id:2d}")
                            bar.total = total_segments
                            
                            # Update progress
                            bar.n = segment
                            
                            # Update postfix with timing info
                            batch_info = f"Batch {batch_num}/{total_batches} | " \
                                         f"Time: {batch_elapsed:.1f}s | ETA: {batch_eta}"
                            bar.set_postfix_str(batch_info)
                            
                            bar.refresh()
            
            # Sleep briefly to reduce CPU usage
            time.sleep(0.2)
            
    finally:
        # Clean up
        for bar in process_bars:
            bar.close()
        master_bar.close()
        sys.stdout.write("\033[?25h")  # Show cursor


if __name__ == "__main__":
    start_time = time.time()
    
    # Load text
    print(f"Loading text from {FILE_NAME}...")
    with open(FILE_NAME, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Chunk text (now with progress tracking)
    print("Chunking text...")
    text_phoneme_chunks = custom_chunk_text(text, LANG_CODE)
    total_chunks = len(text_phoneme_chunks)
    print(f"Text chunking complete: {total_chunks} chunks created")
    
    # Prepare batches for processing
    print("Preparing batches for processing...")
    batches = []
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = [(chunk, VOICE_TO_USE) for chunk in text_phoneme_chunks[i:i+BATCH_SIZE]]
        batches.append(batch)
    
    total_batches = len(batches)
    print(f"Created {total_batches} batches for processing")
    
    # Configure process pool - optimize CPU/GPU usage
    if USE_GPU:
        # When using GPU, limit processes to avoid memory issues
        num_processes = max(4, torch.cuda.device_count())
    else:
        # For CPU, use more processes
        num_processes = max(multiprocessing.cpu_count()-1, 8)
    
    print(f"Using {num_processes} processes for audio generation")
    
    # Create a simple progress tracker with time tracking
    progress_tracker = create_progress_tracker(num_processes, total_batches, total_chunks)
    
    # Initialize process status
    for i in range(num_processes):
        progress_tracker['process_status'][i] = {
            'batch': None,
            'segment': 0,
            'total_segments': 0,
            'status': 'idle'
        }
    
    # Start display thread in the main process
    display_thread = threading.Thread(
        target=display_progress,
        args=(progress_tracker, num_processes, total_batches, total_chunks)
    )
    display_thread.daemon = True
    display_thread.start()
    
    # Create arguments for batch processing with simplified tracker
    process_args = []
    for batch_idx, batch in enumerate(batches):
        # Calculate which process will handle this batch
        process_id = batch_idx % num_processes
        process_args.append((process_id, batch_idx, batch, total_batches, total_chunks, progress_tracker))
    
    # Use imap instead of starmap for better control over execution
    start_time_audio = time.time()
    filepaths_lists = []
    
    # Process batches using a simpler approach
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process one batch at a time to maintain control
            results = pool.imap(process_batch_wrapper, process_args)
            filepaths_lists = list(results)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
    finally:
        # Signal the display thread to stop
        progress_tracker['stop_flag'].value = True
        display_thread.join(timeout=1)
    
    audio_gen_time = time.time() - start_time_audio
    
    # Flatten list of filepaths for easier tracking
    all_filepaths = [fp for sublist in filepaths_lists for fp in sublist]
    print(f"\nAudio generation complete!")
    print(f"Generated {len(all_filepaths)} audio files in {audio_gen_time:.2f} seconds")
    print(f"Average time per chunk: {audio_gen_time / total_chunks:.2f} seconds")
    
    # Join audio segments using streaming approach
    join_audio_segments_streaming()
    
    # Report total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processed {total_chunks} chunks")