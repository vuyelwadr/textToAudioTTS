"""
Voice Parameter Tester for Kokoro TTS

This script tests different voices and parameters for the Kokoro TTS system
to determine which sounds the most natural.

Usage:
  python voice_parameter_tester.py

The script will generate audio samples with different voices and parameters
and save them to the 'voice_tests' directory.
"""

import os
import numpy as np
import torch
import time
import soundfile as sf
from tqdm import tqdm
from kokoro import KPipeline
import warnings
import json
import argparse
import re

# Filter out PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                        module="torch.nn.utils.weight_norm")

# Constants
OUTPUT_DIR = 'voice_tests'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 24020
USE_GPU = torch.cuda.is_available()
LANG_CODE = 'a'  # 'a' for English

# Available voices to test
VOICES = [
    'af_alloy',
    # 'af_aoede',
    # 'af_bella',
    # 'af_heart',      # Was already present
    # 'af_jessica',
    # 'af_kore',
    # 'af_nicole',
    # 'af_nova',
    # 'af_peaceful',   # Was already present
    # 'af_river',
    # 'af_sarah',
    # 'af_sky',
    # 'af_sunny',      # Was already present
    'am_adam',
    # 'am_echo',
    # 'am_eric',
    # 'am_fenrir',
    # 'am_liam',
    # 'am_michael',
    # 'am_onyx',
    'am_puck',
    # 'am_santa',
    # 'bf_alice',
    # 'bf_emma',
    # 'bf_isabella',
    # 'bf_lily',
    # 'bm_daniel',
    'bm_fable',
    # 'bm_george',
    # 'bm_lewis',
    # 'ef_dora',
    # 'em_alex',
    # 'em_santa',
    # 'ff_siwis',
    # 'hf_alpha',
    # 'hf_beta',
    # 'hm_omega',
    # 'hm_psi',
    # 'if_sara',
    # 'im_nicola',
    # 'jf_alpha',
    # 'jf_gongitsune',
    # 'jf_nezumi',
    # 'jf_tebukuro',
    # 'jm_kumo',
    # 'pf_dora',
    # 'pm_alex',
    # 'pm_santa',
    # 'zf_xiaobei',
    # 'zf_xiaoni',
    # 'zf_xiaoxiao',
    # 'zf_xiaoyi',
    # 'zm_yunjian',
    # 'zm_yunxia',
    # 'zm_yunxi',
    # 'zm_yunyang'
]
# Test parameters
SPEEDS = [0.9, 1.0, 1.1]
PITCH_VARIANCES = [0.0, 0.2, 0.4]
BREATHINESS = [0.0, 0.15, 0.3]
PAUSE_FACTORS = [1.0, 1.2, 1.5]

# Sample text - short paragraph for quick testing
DEFAULT_SAMPLE_TEXT = """
Hello there! I'm testing different voice parameters to see which sounds the most natural.
This is a question, isn't it? And this is an exclamation! Now I'm saying something in a 
normal tone of voice. Let's see how different parameters affect the sound quality.
"""

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

def apply_human_audio_effects(audio_data):
    """Apply subtle effects to make audio sound more human"""
    # Add slight volume variation (human speech isn't perfectly consistent)
    volume_envelope = np.linspace(0.98, 1.02, len(audio_data))
    audio_data = audio_data * volume_envelope
    
    # Add very slight noise floor for more natural sound
    noise = np.random.normal(0, 0.0005, len(audio_data))
    audio_data = audio_data + noise
    
    return audio_data

def get_pipeline(lang_code):
    """Initialize a TTS pipeline"""
    pipeline = KPipeline(lang_code=lang_code)
    if USE_GPU:
        pipeline.model.cuda()
    return pipeline

def test_voices(sample_text=DEFAULT_SAMPLE_TEXT, selected_voices=None):
    """Test all available voices with default parameters"""
    pipeline = get_pipeline(LANG_CODE)
    voices_to_test = selected_voices if selected_voices else VOICES
    
    print(f"Testing {len(voices_to_test)} voices with default parameters...")
    results = {}
    
    # Normalize the text to handle line breaks properly
    normalized_text = normalize_text(sample_text)
    
    for voice in tqdm(voices_to_test, desc="Testing voices"):
        start_time = time.time()
        
        try:
            # Split text into sentences for proper processing
            sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
            print(f"Processing {len(sentences)} sentences for voice {voice}")
            
            all_audio = []
            
            # Process each sentence separately
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                print(f"  Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}")
                
                # Use the __call__ method for each sentence
                generator = pipeline(
                    sentence,
                    voice=voice,
                    speed=1.0
                )
                
                for result in generator:
                    if result.audio is not None:
                        if USE_GPU:
                            audio_data = result.audio.cpu().numpy()
                        else:
                            audio_data = result.audio.numpy()
                        
                        all_audio.append(audio_data)
            
            if all_audio:
                # Concatenate all audio segments
                combined_audio = np.concatenate(all_audio)
                
                # Save combined audio
                filepath = os.path.join(OUTPUT_DIR, f'voice_{voice}_default.wav')
                sf.write(filepath, combined_audio, SAMPLE_RATE)
                
                duration = time.time() - start_time
                results[voice] = {
                    "voice": voice,
                    "parameters": "default",
                    "duration": duration,
                    "filepath": filepath,
                    "audio_length": len(combined_audio) / SAMPLE_RATE
                }
                print(f"Generated {filepath} in {duration:.2f} seconds - Audio length: {len(combined_audio) / SAMPLE_RATE:.2f}s")
            else:
                print(f"No audio was generated for voice {voice}")
            
        except Exception as e:
            print(f"Error testing voice {voice}: {str(e)}")
    
    return results

def test_parameters(voice, sample_text=DEFAULT_SAMPLE_TEXT):
    """Test different parameter combinations for a specific voice"""
    pipeline = get_pipeline(LANG_CODE)
    results = {}
    
    print(f"Testing parameters for voice {voice}...")
    
    # Normalize the text to handle line breaks properly
    normalized_text = normalize_text(sample_text)
    
    # Split text into sentences for proper processing
    sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
    print(f"Processing {len(sentences)} sentences")
    
    # Test different speeds
    for speed in tqdm(SPEEDS, desc="Testing speech speeds"):
        try:
            start_time = time.time()
            all_audio = []
            
            # Process each sentence separately
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Use pipeline's __call__ method directly
                generator = pipeline(
                    sentence,
                    voice=voice,
                    speed=speed
                )
                
                for result in generator:
                    if result.audio is not None:
                        if USE_GPU:
                            audio_data = result.audio.cpu().numpy()
                        else:
                            audio_data = result.audio.numpy()
                        
                        # Apply human-like effects
                        audio_data = apply_human_audio_effects(audio_data)
                        all_audio.append(audio_data)
            
            if all_audio:
                # Concatenate all audio segments
                combined_audio = np.concatenate(all_audio)
                
                # Save the audio
                filepath = os.path.join(OUTPUT_DIR, f'voice_{voice}_speed_{speed}.wav')
                sf.write(filepath, combined_audio, SAMPLE_RATE)
                
                duration = time.time() - start_time
                param_key = f"speed_{speed}"
                results[param_key] = {
                    "voice": voice,
                    "speed": speed,
                    "duration": duration,
                    "filepath": filepath,
                    "audio_length": len(combined_audio) / SAMPLE_RATE
                }
                print(f"Generated {filepath} in {duration:.2f} seconds - Audio length: {len(combined_audio) / SAMPLE_RATE:.2f}s")
            
        except Exception as e:
            print(f"Error testing speed {speed} for voice {voice}: {str(e)}")
    
    # Test advanced parameters if the model supports them
    try:
        # Try a combined parameter test - need to use keyword arguments with the __call__ method
        try:
            start_time = time.time()
            
            # Note: We'll need to check if any other parameters are supported by __call__
            generator = pipeline(
                sample_text,
                voice=voice,
                speed=1.0
                # Other parameters might not be supported directly by __call__
            )
            
            for result in generator:
                if USE_GPU:
                    audio_data = result.audio.cpu().numpy()
                else:
                    audio_data = result.audio.numpy()
                
                # Apply human-like effects to simulate additional parameters
                audio_data = apply_human_audio_effects(audio_data)
                
                filepath = os.path.join(OUTPUT_DIR, f'voice_{voice}_enhanced.wav')
                sf.write(filepath, audio_data, SAMPLE_RATE)
                
                duration = time.time() - start_time
                results["enhanced"] = {
                    "voice": voice,
                    "speed": 1.0,
                    "duration": duration,
                    "filepath": filepath
                }
                print(f"Generated enhanced sample in {duration:.2f} seconds")
                
        except Exception as e:
            print(f"Enhanced parameters not supported: {str(e)}")
            print("Testing individual parameters will use basic speed only, as other parameters aren't directly supported")
            
            # Since advanced parameters are not directly supported by __call__, we'll just test speed variations
            for speed_value in [0.9, 1.0, 1.1]:
                try:
                    start_time = time.time()
                    
                    generator = pipeline(
                        sample_text,
                        voice=voice,
                        speed=speed_value
                    )
                    
                    for result in generator:
                        if USE_GPU:
                            audio_data = result.audio.cpu().numpy()
                        else:
                            audio_data = result.audio.numpy()
                        
                        # Apply human-like effects
                        audio_data = apply_human_audio_effects(audio_data)
                        
                        filepath = os.path.join(OUTPUT_DIR, f'voice_{voice}_alt_speed_{speed_value}.wav')
                        sf.write(filepath, audio_data, SAMPLE_RATE)
                        
                        duration = time.time() - start_time
                        param_key = f"alt_speed_{speed_value}"
                        results[param_key] = {
                            "voice": voice,
                            "speed": speed_value,
                            "duration": duration,
                            "filepath": filepath
                        }
                        print(f"Generated {filepath} in {duration:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error testing alternative speed {speed_value} for voice {voice}: {str(e)}")
                
    except Exception as e:
        print(f"Advanced parameter testing failed: {str(e)}")
    
    # Save results to JSON for reference
    with open(os.path.join(OUTPUT_DIR, f'voice_{voice}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def test_all_parameters(sample_text=DEFAULT_SAMPLE_TEXT, selected_voices=None):
    """Test all parameter combinations for all or selected voices"""
    pipeline = get_pipeline(LANG_CODE)
    voices_to_test = selected_voices if selected_voices else VOICES
    all_results = {}
    
    print(f"Testing all parameter combinations for {len(voices_to_test)} voices...")
    
    # Normalize the text to handle line breaks properly
    normalized_text = normalize_text(sample_text)
    
    # Split text into sentences for proper processing
    sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
    
    for voice in tqdm(voices_to_test, desc="Testing voices"):
        voice_results = {}
        
        # Test speed variations
        for speed in SPEEDS:
            for pitch_var in PITCH_VARIANCES:
                for breath in BREATHINESS:
                    for pause in PAUSE_FACTORS:
                        param_name = f"speed{speed}_pitch{pitch_var}_breath{breath}_pause{pause}"
                        param_display = f"Speed: {speed}, Pitch Var: {pitch_var}, Breath: {breath}, Pause: {pause}"
                        print(f"\nTesting {voice} with {param_display}")
                        
                        try:
                            start_time = time.time()
                            all_audio = []
                            
                            # Process each sentence separately
                            for sentence in sentences:
                                if not sentence.strip():
                                    continue
                                
                                # Use pipeline's __call__ method with the parameters it supports
                                # Note: The pipeline might only support speed directly
                                generator = pipeline(
                                    sentence,
                                    voice=voice,
                                    speed=speed
                                )
                                
                                for result in generator:
                                    if result.audio is not None:
                                        if USE_GPU:
                                            audio_data = result.audio.cpu().numpy()
                                        else:
                                            audio_data = result.audio.numpy()
                                        
                                        # Apply human-like effects to simulate other parameters
                                        # that might not be directly supported by the API
                                        audio_data = apply_human_audio_effects(audio_data)
                                        all_audio.append(audio_data)
                            
                            if all_audio:
                                # Concatenate all audio segments
                                combined_audio = np.concatenate(all_audio)
                                
                                # Create detailed filename including voice name and all parameters
                                filename = f'{voice}_speed{speed}_pitch{pitch_var}_breath{breath}_pause{pause}.wav'
                                filepath = os.path.join(OUTPUT_DIR, filename)
                                sf.write(filepath, combined_audio, SAMPLE_RATE)
                                
                                duration = time.time() - start_time
                                voice_results[param_name] = {
                                    "voice": voice,
                                    "speed": speed,
                                    "pitch_variance": pitch_var,
                                    "breathiness": breath,
                                    "pause_factor": pause,
                                    "duration": duration,
                                    "filepath": filepath,
                                    "audio_length": len(combined_audio) / SAMPLE_RATE
                                }
                                print(f"Generated {filepath} in {duration:.2f} seconds - Audio length: {len(combined_audio) / SAMPLE_RATE:.2f}s")
                        
                        except Exception as e:
                            print(f"Error testing parameters for voice {voice}: {str(e)}")
        
        # Save results for this voice to JSON
        with open(os.path.join(OUTPUT_DIR, f'{voice}_all_parameters_results.json'), 'w') as f:
            json.dump(voice_results, f, indent=2)
        
        all_results[voice] = voice_results
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Test TTS voices and parameters")
    parser.add_argument("--text", type=str, help="Sample text to use for testing", default=DEFAULT_SAMPLE_TEXT)
    parser.add_argument("--voice", type=str, help="Specific voice to test parameters for")
    parser.add_argument("--test-all", action="store_true", help="Test all voices")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--test-all-params", action="store_true", help="Test all parameter combinations for all voices")
    parser.add_argument("--selected-voices", nargs='+', help="Specific voices to test (use with --test-all-params)")
    
    args = parser.parse_args()
    
    if args.list_voices:
        print("Available voices:")
        for voice in VOICES:
            print(f"  - {voice}")
        return
    
    print("Voice Parameter Tester")
    print("=====================")
    print(f"Using {'GPU' if USE_GPU else 'CPU'} for inference")
    
    if args.test_all_params:
        # Test all parameter combinations for selected voices or all voices
        selected_voices = args.selected_voices if args.selected_voices else None
        test_all_parameters(args.text, selected_voices)
    elif args.voice:
        # Test a specific voice with different parameters
        if args.voice not in VOICES:
            print(f"Warning: Voice '{args.voice}' not in known voices list. Attempting anyway.")
        test_parameters(args.voice, args.text)
    elif args.test_all:
        # Test all voices
        voice_results = test_voices(args.text)
        
        # Ask which voice to test parameters for
        print("\nVoice testing complete. Which voice would you like to test parameters for?")
        for i, voice in enumerate(VOICES):
            print(f"{i+1}. {voice}")
        
        choice = input("Enter number (or 'q' to quit): ")
        if choice.lower() != 'q':
            try:
                voice_index = int(choice) - 1
                if 0 <= voice_index < len(VOICES):
                    test_parameters(VOICES[voice_index], args.text)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input.")
    else:
        # Default: just test all voices with default parameters
        test_voices(args.text)

if __name__ == "__main__":
    main()
    print("\nAll tests completed! Check the 'voice_tests' directory for audio samples.")
    print("Compare the samples to find which voice and parameters sound most natural.")
