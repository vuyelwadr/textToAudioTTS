#!/usr/bin/env python3
"""
Web TTS Integration Script
Demonstrates how to use the text chunking adapter with web TTS automation
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from textChunkingAdapter import WebTTSChunker

REPO_ROOT = Path(__file__).resolve().parents[1]
INDABA_DIR = REPO_ROOT / "library" / "credoMutwa" / "indaba"
DEFAULT_TTS_DIR = INDABA_DIR / "ttsOnline"
DEFAULT_INPUT_FILE = INDABA_DIR / "indaba1.1.txt"

class WebTTSAutomation:
    """
    Manages automated web TTS processing using the chunking adapter
    """
    
    def __init__(self, 
                 tts_dir: str,
                 num_workers: int = 4,
                 delay_between_requests: float = 2.0):
        """
        Initialize the web TTS automation system
        
        Args:
            tts_dir: Directory containing chunks and manifest
            num_workers: Number of parallel browser instances
            delay_between_requests: Delay between TTS requests (seconds)
        """
        self.tts_dir = tts_dir
        self.num_workers = num_workers
        self.delay_between_requests = delay_between_requests
        self.progress_file = os.path.join(tts_dir, "progress.json")
        self.audio_output_dir = os.path.join(tts_dir, "audio")
        
        # Create audio output directory
        os.makedirs(self.audio_output_dir, exist_ok=True)
        
        # Load manifest and processing plan
        self.manifest = self._load_manifest()
        self.processing_plan = self._load_processing_plan()
        self.progress = self._load_progress()
        
    def _load_manifest(self) -> Dict:
        """Load the manifest file"""
        manifest_path = os.path.join(self.tts_dir, "indaba1.1_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_processing_plan(self) -> Dict:
        """Load the processing plan file"""
        plan_path = os.path.join(self.tts_dir, "processing_plan.json")
        if os.path.exists(plan_path):
            with open(plan_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_progress(self) -> Dict:
        """Load or create progress tracking"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Initialize progress
        return {
            'completed_chunks': [],
            'failed_chunks': [],
            'in_progress': [],
            'start_time': None,
            'end_time': None,
            'worker_status': {}
        }
    
    def _save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def _read_chunk_text(self, filename: str) -> str:
        """Read text from a chunk file"""
        chunk_path = os.path.join(self.tts_dir, filename)
        with open(chunk_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _simulate_web_tts_request(self, chunk_id: int, text: str) -> bool:
        """
        Simulate a web TTS request
        In a real implementation, this would use selenium/playwright to interact with the web TTS
        
        Args:
            chunk_id: ID of the chunk
            text: Text to convert to speech
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Simulate processing time (web TTS typically takes 2-5 seconds per chunk)
            processing_time = 2.0 + (len(text) / 3000) * 3.0  # Scale with text length
            time.sleep(processing_time)
            
            # Simulate audio file creation
            audio_filename = f"chunk_{chunk_id:04d}.mp3"
            audio_path = os.path.join(self.audio_output_dir, audio_filename)
            
            # Create a dummy audio file (in real implementation, this would be downloaded)
            with open(audio_path, 'w') as f:
                f.write(f"# Simulated audio file for chunk {chunk_id}\n")
                f.write(f"# Text length: {len(text)} characters\n")
                f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            
            # Simulate occasional failures (5% failure rate for demo)
            import random
            if random.random() < 0.05:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")
            return False
    
    def _process_worker_chunk(self, worker_id: int, chunk_info: Dict) -> bool:
        """
        Process a single chunk as a worker
        
        Args:
            worker_id: ID of the worker
            chunk_info: Chunk information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        chunk_id = chunk_info['chunk_id']
        filename = chunk_info['filename']
        
        # Skip if already completed
        if chunk_id in self.progress['completed_chunks']:
            print(f"Worker {worker_id}: Chunk {chunk_id} already completed, skipping")
            return True
        
        # Mark as in progress
        if chunk_id not in self.progress['in_progress']:
            self.progress['in_progress'].append(chunk_id)
        self.progress['worker_status'][worker_id] = {
            'current_chunk': chunk_id,
            'status': 'processing',
            'timestamp': datetime.now().isoformat()
        }
        self._save_progress()
        
        print(f"Worker {worker_id}: Processing chunk {chunk_id} ({filename})")
        
        # Read chunk text
        text = self._read_chunk_text(filename)
        
        # Simulate web TTS request
        success = self._simulate_web_tts_request(chunk_id, text)
        
        # Update progress
        if success:
            self.progress['completed_chunks'].append(chunk_id)
            if chunk_id in self.progress['in_progress']:
                self.progress['in_progress'].remove(chunk_id)
            print(f"Worker {worker_id}: ✅ Completed chunk {chunk_id}")
        else:
            self.progress['failed_chunks'].append(chunk_id)
            if chunk_id in self.progress['in_progress']:
                self.progress['in_progress'].remove(chunk_id)
            print(f"Worker {worker_id}: ❌ Failed chunk {chunk_id}")
        
        self.progress['worker_status'][worker_id]['status'] = 'idle' if success else 'error'
        self._save_progress()
        
        return success
    
    def _worker_thread(self, worker_id: int, chunk_list: List[Dict]):
        """
        Worker thread function for processing chunks
        
        Args:
            worker_id: ID of the worker thread
            chunk_list: List of chunks to process
        """
        print(f"Worker {worker_id} started with {len(chunk_list)} chunks")
        
        for chunk_info in chunk_list:
            try:
                # Process the chunk
                self._process_worker_chunk(worker_id, chunk_info)
                
                # Delay between requests
                if self.delay_between_requests > 0:
                    time.sleep(self.delay_between_requests)
                    
            except Exception as e:
                print(f"Worker {worker_id} error: {str(e)}")
                continue
        
        print(f"Worker {worker_id} finished")
    
    def start_processing(self) -> bool:
        """
        Start the parallel TTS processing
        
        Returns:
            True if processing started successfully
        """
        if not self.manifest or not self.processing_plan:
            print("❌ No manifest or processing plan found. Run chunking first.")
            return False
        
        total_chunks = self.manifest['metadata']['total_chunks']
        completed_chunks = len(self.progress['completed_chunks'])
        
        print(f"🎤 Starting Web TTS Processing")
        print(f"📊 Total chunks: {total_chunks}")
        print(f"✅ Completed: {completed_chunks}")
        print(f"⏳ Remaining: {total_chunks - completed_chunks}")
        print(f"👥 Workers: {self.num_workers}")
        
        if completed_chunks >= total_chunks:
            print("✅ All chunks already completed!")
            return True
        
        # Set start time if not set
        if not self.progress['start_time']:
            self.progress['start_time'] = datetime.now().isoformat()
            self._save_progress()
        
        # Get chunks that need processing
        all_chunks = self.manifest['chunks']
        pending_chunks = [c for c in all_chunks if c['chunk_id'] not in self.progress['completed_chunks']]
        
        # Distribute chunks among workers
        workers = []
        for worker_info in self.processing_plan['workers'][:self.num_workers]:
            worker_id = worker_info['worker_id']
            chunk_range = worker_info['chunk_range']
            
            # Filter chunks for this worker that are still pending
            worker_chunks = [c for c in pending_chunks 
                           if chunk_range[0] <= c['chunk_id'] <= chunk_range[1]]
            
            if worker_chunks:
                thread = threading.Thread(
                    target=self._worker_thread,
                    args=(worker_id, worker_chunks)
                )
                workers.append(thread)
                thread.start()
        
        # Wait for all workers to complete
        for thread in workers:
            thread.join()
        
        # Set end time
        self.progress['end_time'] = datetime.now().isoformat()
        self._save_progress()
        
        # Print final status
        self._print_final_status()
        
        return True
    
    def _print_final_status(self):
        """Print final processing status"""
        total_chunks = self.manifest['metadata']['total_chunks']
        completed = len(self.progress['completed_chunks'])
        failed = len(self.progress['failed_chunks'])
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Total chunks: {total_chunks}")
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success rate: {(completed/total_chunks*100):.1f}%")
        
        if self.progress['start_time'] and self.progress['end_time']:
            start = datetime.fromisoformat(self.progress['start_time'])
            end = datetime.fromisoformat(self.progress['end_time'])
            duration = end - start
            print(f"⏱️  Total time: {duration}")
        
        print(f"📁 Audio files saved to: {self.audio_output_dir}")
    
    def get_status(self) -> Dict:
        """Get current processing status"""
        if not self.manifest:
            return {'error': 'No manifest found'}
        
        total_chunks = self.manifest['metadata']['total_chunks']
        completed = len(self.progress['completed_chunks'])
        failed = len(self.progress['failed_chunks'])
        in_progress = len(self.progress['in_progress'])
        
        status = {
            'total_chunks': total_chunks,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'remaining': total_chunks - completed - failed,
            'progress_percentage': (completed / total_chunks * 100) if total_chunks > 0 else 0
        }
        
        if self.progress['start_time']:
            start = datetime.fromisoformat(self.progress['start_time'])
            if self.progress['end_time']:
                end = datetime.fromisoformat(self.progress['end_time'])
                status['elapsed_time'] = str(end - start)
                status['is_running'] = False
            else:
                now = datetime.now()
                status['elapsed_time'] = str(now - start)
                status['is_running'] = True
        
        return status

def main():
    """
    Main function demonstrating the web TTS integration
    """
    print("🎤 Web TTS Integration Demo")
    print("=" * 50)
    
    # Configuration
    tts_dir = str(DEFAULT_TTS_DIR)
    
    # Check if chunks exist
    manifest_path = os.path.join(tts_dir, "indaba1.1_manifest.json")
    if not os.path.exists(manifest_path):
        print("❌ No manifest found. Running chunking first...")
        
        # Run chunking
        input_file = str(DEFAULT_INPUT_FILE)
        chunker = WebTTSChunker(
            max_chunk_size=6000,
            min_chunk_size=500,
            output_dir=tts_dir
        )
        
        manifest_path, plan_path = chunker.process_file(
            input_file=input_file,
            prefix="indaba1.1",
            num_workers=4
        )
        
        print("✅ Chunking completed!")
        print()
    
    # Initialize automation
    automation = WebTTSAutomation(
        tts_dir=tts_dir,
        num_workers=4,
        delay_between_requests=1.0  # Short delay for demo
    )
    
    # Get initial status
    status = automation.get_status()
    print(f"📊 Current Status: {status['completed']}/{status['total_chunks']} chunks completed ({status['progress_percentage']:.1f}%)")
    
    # Start processing
    if status['remaining'] > 0:
        print(f"\n🚀 Starting processing with {automation.num_workers} workers...")
        success = automation.start_processing()
        
        if success:
            final_status = automation.get_status()
            print(f"\n📊 Final Status: {final_status['completed']}/{final_status['total_chunks']} chunks completed ({final_status['progress_percentage']:.1f}%)")
        else:
            print("❌ Processing failed!")
    else:
        print("✅ All chunks already processed!")
    
    print(f"\n📁 Check the output directory: {tts_dir}")
    print(f"🎵 Audio files: {os.path.join(tts_dir, 'audio')}")
    print(f"📋 Progress tracking: {os.path.join(tts_dir, 'progress.json')}")

if __name__ == "__main__":
    main()
