# Text Chunking Adapter for Web TTS

This adapter provides optimized text chunking for web-based Text-to-Speech (TTS) services with large character limits, designed to work with parallel browser automation.

## Overview

The `textChunkingAdapter.py` script chunks large text files into optimized segments for web TTS processing, with features for:
- Sentence boundary preservation for natural speech
- Optimized chunk sizes for web TTS (6000+ characters)
- Parallel processing coordination
- Progress tracking and manifest generation

## Key Features

### 1. Smart Text Chunking
- **Optimized for Web TTS**: Uses 6000-character chunks (web TTS can handle large texts)
- **Natural Speech Boundaries**: Respects sentence and clause boundaries
- **Intelligent Splitting**: Prioritizes sentence endings over clause separators
- **Content Hashing**: Generates unique hashes for each chunk for verification

### 2. Parallel Processing Ready
- **Worker Distribution**: Automatically divides chunks among multiple workers
- **Processing Plans**: Generates detailed plans for parallel browser instances
- **Progress Tracking**: Real-time progress monitoring with JSON persistence
- **Conflict Prevention**: Designed to avoid race conditions in parallel processing

### 3. Comprehensive Metadata
- **Chunk Manifest**: Detailed information about each chunk (size, duration, hash)
- **Processing Plans**: Worker assignments and chunk distribution
- **Progress Files**: Persistent tracking of completed/failed chunks
- **Audio Integration**: Ready for audio file management

## File Structure

```
ttsOnline/
├── chunk_0000_hash.txt          # Individual text chunks
├── chunk_0001_hash.txt
├── ...
├── indaba1.1_manifest.json      # Chunk metadata and manifest
├── processing_plan.json         # Worker distribution plan
├── progress.json               # Processing progress tracking
└── audio/                      # Generated audio files
    ├── chunk_0000.mp3
    ├── chunk_0001.mp3
    └── ...
```

## Usage

### Basic Chunking

```python
from textChunkingAdapter import WebTTSChunker

# Initialize chunker
chunker = WebTTSChunker(
    max_chunk_size=6000,  # Optimized for web TTS
    min_chunk_size=500,
    output_dir="ttsOnline"
)

# Process a file
manifest_path, plan_path = chunker.process_file(
    input_file="input.txt",
    prefix="document",
    num_workers=6  # Plan for 6 parallel workers
)
```

### Command Line Usage

```bash
# Run the adapter directly
python textChunkingAdapter.py

# Or use the test script
python test_chunking_adapter.py
```

### Web TTS Integration

```python
from webTTS_integration import WebTTSAutomation

# Initialize automation
automation = WebTTSAutomation(
    tts_dir="ttsOnline",
    num_workers=4,
    delay_between_requests=2.0
)

# Start parallel processing
automation.start_processing()

# Check status
status = automation.get_status()
print(f"Progress: {status['progress_percentage']:.1f}%")
```

## Configuration Options

### WebTTSChunker Parameters

- **max_chunk_size**: Maximum characters per chunk (default: 6000)
- **min_chunk_size**: Minimum characters per chunk (default: 500)
- **output_dir**: Directory for chunks and metadata (default: "ttsOnline")

### WebTTSAutomation Parameters

- **num_workers**: Number of parallel browser instances (default: 4)
- **delay_between_requests**: Delay between TTS requests in seconds (default: 2.0)

## Output Files

### Manifest (indaba1.1_manifest.json)

```json
{
  "metadata": {
    "total_chunks": 25,
    "total_characters": 145832,
    "total_words": 25467,
    "estimated_total_duration_seconds": 10186,
    "max_chunk_size": 6000,
    "created_at": "2024-01-15T10:30:00",
    "source_file": "indaba1.1.txt"
  },
  "chunks": [
    {
      "chunk_id": 0,
      "filename": "chunk_0000_a1b2c3d4.txt",
      "character_count": 5987,
      "word_count": 1042,
      "estimated_duration_seconds": 416,
      "content_hash": "a1b2c3d4"
    }
  ]
}
```

### Processing Plan (processing_plan.json)

```json
{
  "metadata": {
    "num_workers": 6,
    "total_chunks": 25,
    "chunks_per_worker": 5
  },
  "workers": [
    {
      "worker_id": 0,
      "chunk_range": [0, 4],
      "num_chunks": 5,
      "chunks": ["chunk_0000_a1b2c3d4.txt", ...]
    }
  ]
}
```

### Progress Tracking (progress.json)

```json
{
  "completed_chunks": [0, 1, 2],
  "failed_chunks": [],
  "in_progress": [3],
  "start_time": "2024-01-15T10:30:00",
  "worker_status": {
    "0": {
      "current_chunk": 3,
      "status": "processing",
      "timestamp": "2024-01-15T10:35:00"
    }
  }
}
```

## Integration with Web TTS Services

The adapter is designed to work with any web TTS service that can handle large text inputs. Here's how to integrate:

### 1. Chunk Preparation
```python
# Use the adapter to create optimized chunks
chunker = WebTTSChunker(max_chunk_size=6000)
manifest_path, plan_path = chunker.process_file("large_text.txt")
```

### 2. Browser Automation Setup
```python
# Each worker gets its assigned chunks
automation = WebTTSAutomation(num_workers=6)
```

### 3. Parallel Processing
- Each worker processes its assigned chunk range
- Workers avoid conflicts by using assigned chunk IDs
- Progress is tracked centrally to prevent duplicate work

### 4. Audio Collection
- Audio files are saved with chunk IDs for proper ordering
- Manifest provides metadata for final assembly
- Progress tracking ensures all chunks are processed

## Advantages Over Simple Splitting

1. **Natural Speech**: Preserves sentence boundaries for better TTS quality
2. **Optimized Sizing**: Uses web TTS's large character limits efficiently
3. **Parallel Ready**: Designed for multi-browser automation
4. **Progress Tracking**: Persistent progress for long-running processes
5. **Error Recovery**: Failed chunks can be retried individually
6. **Verification**: Content hashes ensure text integrity

## Error Handling

- **Missing Files**: Graceful handling of missing input files
- **Chunk Validation**: Content hashes verify text integrity
- **Progress Recovery**: Can resume from interruptions
- **Worker Failures**: Individual worker failures don't stop others

## Performance Considerations

- **Memory Usage**: Processes text in streams for large files
- **Parallel Efficiency**: Balanced chunk distribution for optimal performance
- **Request Throttling**: Configurable delays to avoid overwhelming TTS services
- **Progress Persistence**: Minimal I/O overhead for progress tracking

## Example Workflow

1. **Setup**: Initialize chunker with desired parameters
2. **Chunking**: Process large text file into optimized chunks
3. **Planning**: Generate worker distribution plan
4. **Processing**: Start parallel web TTS processing
5. **Monitoring**: Track progress and handle failures
6. **Completion**: Assemble final audio from chunk files

This adapter provides a robust foundation for web TTS automation with large text files, ensuring efficient parallel processing while maintaining audio quality through intelligent text segmentation.
