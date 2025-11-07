# Parallel TTS Downloader - Complete Guide

## ✅ What's Fixed

Your original script failed silently. This new version provides:

1. **Detailed Error Reporting** - Shows EXACTLY which step failed
2. **Step-by-step Validation** - Validates each operation
3. **Better Timeout Handling** - Adaptive waiting based on text length
4. **Concurrent Support** - Download multiple chunks in parallel
5. **Test Mode** - Validate setup before processing large files

## 🚀 Quick Start

### Installation

```bash
# Install Playwright
pip install playwright

# Install browser
python3 -m playwright install chromium
```

### Basic Usage

```bash
# Test with a small file first (recommended)
python3 parallel_tts_downloader.py your_file.txt --test

# Process entire file (sequential, most reliable)
python3 parallel_tts_downloader.py your_file.txt

# Concurrent processing (3 chunks at a time)
python3 parallel_tts_downloader.py your_file.txt --concurrent 3

# Custom chunk size
python3 parallel_tts_downloader.py your_file.txt --chunk-size 2000
```

## 📊 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `input_file` | Required | Path to input text file |
| `--output-dir` | Auto | Output directory for MP3 files |
| `--concurrent N` | 1 | Max concurrent downloads (1=sequential) |
| `--chunk-size N` | 3000 | Characters per chunk |
| `--test` | False | Test mode: only process first 2 chunks |
| `--no-headless` | False | Show browser windows (for debugging) |

## 🔧 Usage Examples

### 1. Test Mode (Recommended First)

```bash
# Test with first 2 chunks to verify everything works
python3 parallel_tts_downloader.py mybook.txt --test
```

### 2. Sequential Mode (Most Reliable)

```bash
# Process one chunk at a time (slowest but most reliable)
python3 parallel_tts_downloader.py mybook.txt
```

### 3. Concurrent Mode (Faster)

```bash
# Download 3 chunks in parallel
python3 parallel_tts_downloader.py mybook.txt --concurrent 3

# Download 5 chunks in parallel (faster but more resource intensive)
python3 parallel_tts_downloader.py mybook.txt --concurrent 5
```

### 4. Custom Output

```bash
# Specify output directory
python3 parallel_tts_downloader.py mybook.txt --output-dir ./audio_output
```

### 5. Debug Mode

```bash
# Show browser windows to see what's happening
python3 parallel_tts_downloader.py mybook.txt --test --no-headless
```

## 📝 Detailed Error Reporting

The script reports errors at each step:

```
[1/10] 🌐 Navigating to website...
[1/10] 🍪 Handling consent...
[1/10] ✅ Consent clicked
[1/10] 🌍 Selecting locale...
[1/10] ✅ Locale: English (South Africa)
[1/10] 🎤 Selecting male voice...
[1/10] ✅ Voice: Microsoft Guy Online (Natural) - Male
[1/10] ⌨️ Entering text (2847 chars)...
[1/10] ✅ Text entered
[1/10] 🔊 Clicking Play...
[1/10] ✅ Play clicked
[1/10] ⏳ Waiting 24s for generation...
[1/10] 🔍 Waiting for Download button...
[1/10] ✅ Download button ready!
[1/10] ⬇️ Starting download...
[1/10] ✅ SUCCESS: segment_00000.mp3 (123.4 KB)
```

If a step fails, you see exactly where and why:

```
[5/10] ❌ Failed at step 'download button ready': Download button not ready after 120s
```

## 🎯 Recommended Settings

### For Testing
```bash
python3 parallel_tts_downloader.py yourfile.txt --test --chunk-size 1000
```
- Small chunks (1000 chars)
- Only processes first 2 chunks
- Validates setup quickly

### For Production (Balanced)
```bash
python3 parallel_tts_downloader.py yourfile.txt --concurrent 3 --chunk-size 3000
```
- 3 parallel downloads (good balance)
- 3000 char chunks (reliable size)
- Faster than sequential, stable

### For Speed (Aggressive)
```bash
python3 parallel_tts_downloader.py yourfile.txt --concurrent 10 --chunk-size 2000
```
- 10 parallel downloads (very fast)
- Smaller chunks (2000 chars)
- May have more failures, but much faster

### For Reliability (Conservative)
```bash
python3 parallel_tts_downloader.py yourfile.txt --concurrent 1 --chunk-size 4000
```
- Sequential (no concurrency)
- Larger chunks (fewer total chunks)
- Slowest but most reliable

## ⚙️ How It Works

### 1. Simple Chunking

The script uses **character-based chunking** (not phoneme-based) for simplicity:

```python
def simple_chunk_text(text: str, chunk_size: int = 3000):
    # Splits at sentence boundaries when possible
    # Falls back to character limit
    # Much simpler than phoneme-based chunking
```

### 2. Step-by-Step Validation

Each download validates 11 steps:

1. **Navigation** - Load the website
2. **Consent** - Handle cookie popup
3. **Locale** - Select English (South Africa)
4. **Voice** - Select male voice
5. **Text Entry** - Enter chunk text
6. **Play Click** - Start TTS generation
7. **Generation Wait** - Wait for audio processing
8. **Download Ready** - Wait for download button
9. **Download** - Click and capture download
10. **Save** - Save to output directory
11. **Verify** - Check file exists and has valid size

If ANY step fails, you see exactly which one and why.

### 3. Concurrent Downloads

When using `--concurrent N`:

```python
# Limit to N simultaneous browsers
semaphore = asyncio.Semaphore(max_concurrent)

# Each chunk gets its own browser instance
# Downloads happen in parallel
# Results are collected when all complete
```

## 📊 Output

### Files Created

```
your_output_dir/
├── segment_00000.mp3  # First chunk
├── segment_00001.mp3  # Second chunk
├── segment_00002.mp3  # Third chunk
└── ...
```

### Summary Report

```
======================================================================
📊 SUMMARY
======================================================================
✅ Successful: 45/50
❌ Failed: 5/50
⏱️ Time: 892.3s
📁 Output: mybook_tts_output

❌ Failed chunks:
   Chunk 12: Failed at step 'audio generation': Timeout exceeded
   Chunk 28: Failed at step 'download button ready': Button not found
   ...
======================================================================
```

## 🐛 Troubleshooting

### Issue: "Executable doesn't exist"

```bash
# Install Playwright browsers
python3 -m playwright install chromium
```

### Issue: "ERR_TUNNEL_CONNECTION_FAILED"

- Check your internet connection
- Check if website is accessible: https://www.text-to-speech.online/
- Try without VPN/proxy

### Issue: Timeouts at "download button ready"

- Text chunks might be too long
- Try smaller chunks: `--chunk-size 2000`
- Try test mode first: `--test`

### Issue: Most chunks fail

- Start with `--concurrent 1` (sequential)
- Use `--test` mode first
- Try `--no-headless` to see what's happening
- Check website hasn't changed structure

### Issue: "Download button not ready after 120s"

- Chunk is too long for the website
- Reduce chunk size: `--chunk-size 1500`
- Some chunks may need manual retry

## 💡 Tips

1. **Always test first**: Use `--test` before processing large files
2. **Start sequential**: Use `--concurrent 1` first, then increase
3. **Monitor failures**: Check which chunks fail and why
4. **Adjust chunk size**: If many timeouts, reduce `--chunk-size`
5. **Resume capability**: Failed chunks can be retried manually

## 🔄 Comparing to Original

| Feature | Original Script | New Script |
|---------|----------------|------------|
| Error Messages | ❌ Generic | ✅ **Step-by-step** |
| Timeout Handling | ❌ Fixed 90s | ✅ **Adaptive** |
| Validation | ❌ Minimal | ✅ **11 checkpoints** |
| Chunking | Complex phoneme | ✅ **Simple char-based** |
| Test Mode | ❌ No | ✅ **Built-in** |
| Debug Mode | ❌ No | ✅ **--no-headless** |
| Progress | ❌ Unclear | ✅ **Real-time** |
| Summary | ❌ Basic | ✅ **Detailed stats** |

## 📚 Example Workflow

```bash
# 1. Test with small file
python3 parallel_tts_downloader.py test.txt --test

# 2. If test works, try full file sequentially
python3 parallel_tts_downloader.py largefile.txt --concurrent 1

# 3. If sequential works, increase concurrency
python3 parallel_tts_downloader.py largefile.txt --concurrent 3

# 4. Monitor output and adjust settings as needed
python3 parallel_tts_downloader.py largefile.txt --concurrent 5 --chunk-size 2500
```

## ✅ Success Criteria

You know it's working when you see:

```
[1/10] ✅ SUCCESS: segment_00000.mp3 (123.4 KB)
```

For each chunk, you should see file size > 1KB (typically 50-200 KB depending on text length).

---

**The script is ready to use!** Start with `--test` mode to validate your setup, then scale up to full processing. 🚀
