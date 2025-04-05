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
import argparse

# Filter out specific PyTorch deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                        module="torch.nn.utils.weight_norm")

# Constants - Moved to top for easier configuration
DEFAULT_OUTPUT_DIR = 'atlantis/output_joined'
DEFAULT_FILE_NAME = 'atlantis/atlantis_join.txt'
BATCH_SIZE = 10  # Process chunks in batches for better performance
MAX_THREADS_PER_PROCESS = 2  # Limit threads per process to avoid memory issues
VOICE_TO_USE = 'af_alloy'  # Default voice
LANG_CODE = 'a'  # 'a' for English
SAMPLE_RATE = 24020
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU

# Available voices to try
AVAILABLE_VOICES = ['af_alloy', 'af_heart', 'af_sunny', 'af_peaceful']

# Human-like speech parameters
PITCH_VARIANCE = 0.0  # Add natural pitch variation
BREATHINESS = 0.0    # Add slight breathiness
PAUSE_FACTOR = 1.0    # Slightly longer pauses at punctuation

def derive_output_dir(input_file_path):
    """Derive an output directory based on the input file path"""
    # Get the directory containing the input file
    input_dir = os.path.dirname(input_file_path)
    
    # Get the filename without extension
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Create an output directory path
    if input_dir:
        return os.path.join(input_dir, f"{base_name}_output")
    else:
        return f"{base_name}_output"

def dynamic_speed(text):
    """Adjust speed based on text content for more natural speech"""
    if '?' in text:
        return 0.95  # Slightly slower for questions
    elif '!' in text:
        return 1.05  # Slightly faster for exclamations
    elif re.search(r'".*"', text) or re.search(r'".*"', text):  # Dialogue detection
        return 1.02  # Slightly faster for dialogue
    # Detect long, complex sentences
    elif len(text.split()) > 20 or text.count(',') > 2:
        return 0.97  # Slow down for complex content
    return 1.0  # Default speed

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

def normalize_text(text):
    """
    Normalize text to handle line breaks more naturally:
    1. Join lines that are part of the same sentence
    2. Preserve paragraph breaks (double newlines)
    3. Remove unnecessary whitespace
    """
    # First, preserve paragraph breaks by temporarily replacing them
    text = text.replace('\n\n', '[PARAGRAPH_BREAK]')
    
    # Join lines that are part of the same sentence
    # Replace single newlines with spaces unless they follow punctuation
    text = re.sub(r'(?<![.!?])\n', ' ', text)
    
    # Restore paragraph breaks
    text = text.replace('[PARAGRAPH_BREAK]', '\n\n')
    
    # Fix any excess whitespace
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def improved_chunk_text(text: str, lang_code: str = 'a') -> List[Tuple[str, str]]:
    """
    Enhanced text chunking that preserves semantic boundaries for better flow
    This is an alternative to custom_chunk_text for more natural speech
    """
    pipeline = KPipeline(lang_code=lang_code, model=False) 
    pipeline.g2p.nlp.max_length = 5000000
    
    print(f"Chunking text using improved algorithm...")
    
    # Normalize the text to handle line breaks properly
    text = normalize_text(text)
    
    # First split at paragraph boundaries
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""
        current_phonemes = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Try adding this sentence to current chunk
            temp_chunk = current_chunk + ('' if not current_chunk else ' ') + sentence
            temp_phonemes = pipeline.g2p(temp_chunk)
            
            # Check if adding this sentence would exceed our limit
            if len(temp_phonemes) > 410 and current_chunk:
                # Store current chunk and start a new one
                chunks.append((current_chunk, current_phonemes))
                current_chunk = sentence
                current_phonemes = pipeline.g2p(sentence)
            else:
                # Keep adding to current chunk
                current_chunk = temp_chunk
                current_phonemes = temp_phonemes
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append((current_chunk, current_phonemes))
    
    print(f"Created {len(chunks)} optimized text chunks")
    return chunks

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

def generate_audio_batch(process_id, batch_index, batch_data, total_batches, total_chunks, output_dir, progress_tracker=None):
    """
    Process a batch of segments at once for better performance
    Uses a simplified progress tracker
    Enhanced with human-like speech parameters
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
            # Calculate dynamic speech parameters based on text content
            speed = dynamic_speed(text_chunk)
            
            # Generate audio - only pass supported parameters 
            # Instead of using generate_from_tokens with unsupported parameters, use the __call__ method
            # which only accepts speed as a parameter, or use generate_from_tokens with only supported parameters
            generator = pipeline(
                text_chunk,  # Use the text chunk directly with the __call__ method
                voice=voice,
                speed=speed
            )
            
            for i, result in enumerate(generator):
                filepath = os.path.join(output_dir, f'segment_{global_segment_index}_{i}.wav')
                # Convert to numpy and save
                if USE_GPU:
                    audio_data = result.audio.cpu().numpy()
                else:
                    audio_data = result.audio.numpy()
                
                # Apply human-like post-processing - this is where we add the effects
                # that would have been handled by the parameters if they were supported
                try:
                    audio_data = apply_human_audio_effects(audio_data)
                except Exception as e:
                    print(f"Warning: Audio effects failed: {e}")
                
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

def apply_human_audio_effects(audio_data):
    """Apply subtle effects to make audio sound more human"""
    try:
        # Add slight volume variation (human speech isn't perfectly consistent)
        volume_envelope = np.linspace(0.98, 1.02, len(audio_data))
        audio_data = audio_data * volume_envelope
        
        # Add very slight noise floor for more natural sound
        noise = np.random.normal(0, 0.0005, len(audio_data))
        audio_data = audio_data + noise
        
        return audio_data
    except Exception as e:
        # Return original audio if processing fails
        print(f"Audio post-processing failed: {e}")
        return audio_data

def process_file(file_path, voice=VOICE_TO_USE, output_dir=None):
    """Process a text file and convert it to audio chunks"""
    try:
        # Determine output directory if not specified
        if output_dir is None:
            output_dir = derive_output_dir(file_path)
        
        # Read the input file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Read {len(text)} characters from {file_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Chunk the text with improved line break handling
        print("Chunking text...")
        chunks = improved_chunk_text(text, LANG_CODE)
        print(f"Created {len(chunks)} text chunks")
        
        # Prepare for batch processing
        batches = []
        current_batch = []
        
        for i, chunk in enumerate(chunks):
            current_batch.append((chunk, voice))
            if len(current_batch) >= BATCH_SIZE or i == len(chunks) - 1:
                batches.append(current_batch)
                current_batch = []
        
        total_batches = len(batches)
        print(f"Created {total_batches} batches for processing")
        
        # Set up multiprocessing
        num_processes = min(multiprocessing.cpu_count(), MAX_THREADS_PER_PROCESS)
        print(f"Using {num_processes} processes for audio generation")
        
        # Create progress tracker
        progress = create_progress_tracker(num_processes, total_batches, len(chunks))
        
        # Set global OUTPUT_DIR for this run
        global OUTPUT_DIR
        OUTPUT_DIR = output_dir
        
        # Process batches with multiprocessing
        all_filepaths = []
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            
            for batch_index, batch in enumerate(batches):
                # Submit batch to process pool with output_dir parameter
                result = pool.apply_async(
                    generate_audio_batch,
                    args=(batch_index % num_processes, batch_index, batch, total_batches, len(chunks), output_dir, progress)
                )
                results.append(result)
            
            # Collect results with progress display
            for result in tqdm(results, desc="Processing audio batches"):
                filepaths = result.get()
                all_filepaths.extend(filepaths)
        
        print(f"Generated {len(all_filepaths)} audio segments")
        
        # Clean up GPU memory
        if USE_GPU:
            torch.cuda.empty_cache()
        gc.collect()
        
        return all_filepaths
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def combine_audio_segments(directory, output_filename="combined_audio.wav"):
    """
    Combine all audio segments in the specified directory into one file.
    
    Args:
        directory (str): Directory containing audio segments
        output_filename (str): Name of the output file (default: combined_audio.wav)
        
    Returns:
        str: Path to the combined audio file if successful, None otherwise
    """
    try:
        print(f"Looking for audio segments in {directory}...")
        # Find all WAV files in the directory
        segment_files = [f for f in os.listdir(directory) if f.endswith('.wav') and 'segment_' in f]
        
        if not segment_files:
            print(f"No audio segments found in {directory}")
            return None
        
        print(f"Found {len(segment_files)} audio segments")
        
        # Get full paths
        filepaths = [os.path.join(directory, f) for f in segment_files]
        
        # Sort filepaths by segment number to ensure correct order
        try:
            # First try with the expected pattern
            filepaths.sort(key=lambda x: int(re.search(r'segment_(\d+)_', x).group(1)))
            print(f"Successfully sorted {len(filepaths)} audio files")
        except (AttributeError, TypeError) as e:
            # Fallback to simple string sort if regex fails
            print(f"Warning: Could not extract segment numbers ({e}), falling back to filename sorting")
            filepaths.sort()
        
        # Print the first few filepaths to verify sorting
        if filepaths:
            print(f"First few files to be combined: {filepaths[:min(3, len(filepaths))]}")
        
        combined_audio = []
        for filepath in tqdm(filepaths, desc="Loading audio segments"):
            try:
                audio, sample_rate = sf.read(filepath)
                combined_audio.append(audio)
                if 'sample_rate' not in locals():
                    sample_rate_to_use = sample_rate
            except Exception as e:
                print(f"Error reading {filepath}: {str(e)}")
        
        # Concatenate all audio segments
        if combined_audio:
            combined_audio = np.concatenate(combined_audio)
            
            # Save combined audio
            combined_filepath = os.path.join(directory, output_filename)
            sf.write(combined_filepath, combined_audio, sample_rate_to_use if 'sample_rate_to_use' in locals() else SAMPLE_RATE)
            print(f"Combined audio saved to: {combined_filepath}")
            return combined_filepath
        else:
            print("Error: No audio segments were successfully loaded for combination.")
            return None
    except Exception as e:
        print(f"Error combining audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert text to speech using Kokoro TTS")
    parser.add_argument("--file", type=str, default=DEFAULT_FILE_NAME,
                        help=f"Input text file (default: {DEFAULT_FILE_NAME})")
    parser.add_argument("--voice", type=str, default=VOICE_TO_USE,
                        help=f"Voice to use (default: {VOICE_TO_USE})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: derived from input file path)")
    parser.add_argument("--combine", action="store_true", default=True,
                        help="Combine all audio segments into one file (default: True)")
    parser.add_argument("--no-combine", dest="combine", action="store_false",
                        help="Don't combine audio segments into one file")
    parser.add_argument("--combine-only", type=str, metavar="DIRECTORY",
                        help="Only combine existing audio segments from the specified directory")
    parser.add_argument("--output-name", type=str, default="combined_audio.wav",
                        help="Name for the combined audio file (default: combined_audio.wav)")
    
    args = parser.parse_args()
    
    # Handle combine-only mode
    if args.combine_only:
        if os.path.isdir(args.combine_only):
            combined_file = combine_audio_segments(args.combine_only, args.output_name)
            if combined_file:
                print(f"Successfully combined audio segments into: {combined_file}")
            else:
                print("Failed to combine audio segments")
            return []
        else:
            print(f"Error: Directory not found: {args.combine_only}")
            return []
    
    # Regular processing mode
    # Derive output directory if not specified
    output_dir = args.output if args.output is not None else derive_output_dir(args.file)
    
    print(f"Starting text-to-speech conversion with Kokoro TTS")
    print(f"Using {'GPU' if USE_GPU else 'CPU'} for inference")
    print(f"Processing file: {args.file}")
    print(f"Using voice: {args.voice}")
    print(f"Output directory: {output_dir}")
    
    # Process the file
    filepaths = process_file(args.file, args.voice, output_dir)
    
    # Combine audio segments if requested
    if args.combine and filepaths:
        combine_audio_segments(output_dir, args.output_name)
    
    print("Text-to-speech conversion completed!")
    return filepaths

if __name__ == "__main__":
    main()

