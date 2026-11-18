# TTS Online Automation Script

Automates text-to-speech generation using [text-to-speech.online](https://www.text-to-speech.online/) with Selenium WebDriver.

## Features

- **Automated TTS Generation**: Navigate website, select settings, input text, and download audio
- **Intelligent Text Chunking**: Automatically splits large texts into optimal chunks (5000 chars each)
- **Stealth Mode**: Anti-bot detection with realistic browser behavior
- **English South Africa Support**: Configured for English South African locale with male voices
- **Progress Tracking**: Real-time progress with ETA, memory usage, and processing speed
- **Error Handling**: Automatic retries with exponential backoff
- **Resume Capability**: Process specific chunk ranges to resume interrupted sessions
- **Session Reports**: Detailed JSON reports with statistics and results

## Requirements

### System Requirements
- Python 3.7+
- Chrome browser
- ChromeDriver

### Python Dependencies
```bash
pip install -r requirements.txt
```

### ChromeDriver Setup

#### macOS (Recommended)
```bash
brew install chromedriver
```

#### Manual Installation
1. Download ChromeDriver from [https://chromedriver.chromium.org/](https://chromedriver.chromium.org/)
2. Add ChromeDriver to your PATH or place in the same directory as the script

## Installation

1. **Install dependencies**:
   ```bash
   cd /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/library/credoMutwa/indaba/ttsOnline
   pip install -r requirements.txt
   ```

2. **Verify setup**:
   ```bash
   python test_setup.py
   ```

3. **Run automation**:
   ```bash
   python ttsOnline_automation.py ../indaba1.1.txt
   ```

## Usage

### Basic Usage

Process entire text file:
```bash
python ttsOnline_automation.py input_file.txt
```

### Advanced Usage

Process chunk range (resume interrupted session):
```bash
python ttsOnline_automation.py input_file.txt --start-chunk 50 --end-chunk 100
```

Custom chunk size:
```bash
python ttsOnline_automation.py input_file.txt --chunk-size 3000
```

Run with visible browser (for debugging):
```bash
python ttsOnline_automation.py input_file.txt --no-headless
```

Custom output directory:
```bash
python ttsOnline_automation.py input_file.txt -o my_output_folder
```

### Command Line Options

- `input_file`: Path to the input text file (required)
- `-o, --output`: Output directory (default: `tts_online_output`)
- `--start-chunk`: Start chunk index (default: 0)
- `--end-chunk`: End chunk index (default: end of file)
- `--no-headless`: Disable headless mode (show browser window)
- `--chunk-size`: Maximum characters per chunk (default: 5000)

## Configuration

### Default Settings

- **Website**: https://www.text-to-speech.online/
- **Locale**: English (South Africa)
- **Voice**: Microsoft Guy Online (Natural) - Male voice
- **Speed**: 1.0 (normal speed)
- **Pitch**: 1.0 (normal pitch)
- **Chunk Size**: 5000 characters (safe margin under 6000 limit)
- **Max Retries**: 3 attempts per chunk
- **Retry Delay**: 5 seconds (increases with each retry)

### Customization

You can modify the constants in `ttsOnline_automation.py`:

```python
DEFAULT_LOCALE = "English (South Africa)"
DEFAULT_VOICE = "Microsoft Guy Online (Natural)"
DEFAULT_SPEED = 1.0
DEFAULT_PITCH = 1.0
CHUNK_SIZE = 5000
MAX_RETRIES = 3
```

## Output Files

The script generates several output files:

1. **Audio Files**: `chunk_0001_1234.mp3` - Individual audio chunks
2. **Session Logs**: `tts_online_log_YYYYMMDD_HHMMSS.txt` - Detailed processing logs
3. **Session Reports**: `session_report_YYYYMMDD_HHMMSS.json` - JSON report with statistics

## Processing the Credo Mutwa Indaba Text

### Full Processing
```bash
cd /Users/ruwodda/Documents/Personal/Repos/textToAudioTTS/library/credoMutwa/indaba/ttsOnline
python ttsOnline_automation.py ../indaba1.1.txt
```

### Resume Processing
If processing is interrupted, check the latest log file for the last successful chunk, then resume:

```bash
# Resume from chunk 25
python ttsOnline_automation.py ../indaba1.1.txt --start-chunk 25
```

### Batch Processing
Process in smaller batches for better reliability:

```bash
# Process first 20 chunks
python ttsOnline_automation.py ../indaba1.1.txt --end-chunk 20

# Process next 20 chunks
python ttsOnline_automation.py ../indaba1.1.txt --start-chunk 20 --end-chunk 40
```

## Troubleshooting

### Common Issues

1. **ChromeDriver not found**
   ```bash
   # On macOS
   brew install chromedriver
   
   # Or add to PATH manually
   export PATH=$PATH:/path/to/chromedriver
   ```

2. **Website blocked/unavailable**
   - Check internet connection
   - Try running with `--no-headless` to see browser behavior
   - Check if website structure has changed

3. **Memory issues**
   - Reduce chunk size: `--chunk-size 3000`
   - Close other applications
   - Process in smaller batches

4. **Rate limiting**
   - The script includes random delays between requests
   - If blocked, wait and resume from last successful chunk

### Debug Mode

Run with visible browser to debug issues:
```bash
python ttsOnline_automation.py input.txt --no-headless
```

### Log Analysis

Check the latest log file for detailed error information:
```bash
tail -f tts_online_output/tts_online_log_*.txt
```

## Performance Tips

1. **Optimal Chunk Size**: 4000-5000 characters provides good balance of speed and reliability
2. **Batch Processing**: Process 20-50 chunks at a time for better error recovery
3. **System Resources**: Close unnecessary applications during processing
4. **Network**: Stable internet connection is crucial for reliable processing

## Integration with Existing Workflow

This script is designed to work alongside existing TTS implementations in the project:

- Compatible with the chunking pattern used in `audioChunkedKukuro.py`
- Outputs can be joined using existing audio joining scripts
- Session reports can be analyzed for quality assessment
- Failed chunks can be retried individually

## Legal and Usage Notes

- Respect the website's terms of service
- Use reasonable request rates to avoid overloading the service
- This tool is for personal/non-commercial use
- Be mindful of copyright when processing copyrighted text

## Support

For issues or questions:

1. Check the log files for detailed error information
2. Run `python test_setup.py` to verify installation
3. Use `--no-headless` mode for visual debugging
4. Check the session reports for processing statistics
