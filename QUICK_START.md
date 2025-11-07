# TTS Download - Quick Start Guide

## ✅ What's Fixed

Your original script had **syntax errors in the JavaScript code**:

1. **Line 84** - Missing closing parenthesis:
   ```javascript
   // WRONG
   if (consentTexts.some(text => button.textContent.includes(text)) {

   // FIXED
   if (consentTexts.some(text => button.textContent.includes(text))) {
   ```

2. **Line 152-165** - String not closed properly:
   ```python
   # WRONG
   text_js = """
   (text) => { ... }

   test_text = "..."  # This was INSIDE the string!

   # FIXED
   text_js = """
   (text) => { ... }
   """  # String closed here

   test_text = "..."  # Now properly outside
   ```

## 🚀 Installation

```bash
# Install Playwright
pip install playwright

# Install browser (only needed once)
playwright install chromium
```

## 📝 Usage

### Simple Test (Single Chunk)

```bash
# Test with visible browser (recommended for first run)
python3 playwright_tts_download.py

# This will:
# - Open browser window
# - Navigate to text-to-speech.online
# - Process a short test sentence
# - Download to debug_download/working_test_download.mp3
```

### Production Script (Multiple Chunks)

```bash
# Process a text file
python3 tts_download_production.py test_input.txt

# With options
python3 tts_download_production.py your_file.txt \
    --output my_output_dir \
    --start-chunk 0 \
    --end-chunk 50 \
    --no-headless
```

## 📂 Files Created

1. **playwright_tts_download.py** - Simple test script (fixed syntax)
   - Single chunk test
   - Shows browser window
   - Downloads to `debug_download/`
   - Good for testing/debugging

2. **tts_download_production.py** - Production script
   - Multiple chunk support
   - Progress tracking
   - Error retry logic
   - Session reports
   - Headless mode option

3. **test_input.txt** - Sample text file for testing

## 🎯 Quick Test Commands

```bash
# 1. Simple test (visible browser)
python3 playwright_tts_download.py

# 2. Production test with sample file
python3 tts_download_production.py test_input.txt --no-headless

# 3. Process large file (headless, faster)
python3 tts_download_production.py your_large_file.txt
```

## 📊 Expected Output

```
🎯 TTS DOWNLOAD TEST
======================================================================
📁 Target: /path/to/debug_download
✅ Playwright setup complete
🌐 Navigating...
🍪 Handling consent...
✅ Consent clicked via JavaScript
✅ Overlays removed
🌍 Selecting English South Africa...
✅ Locale selected via JavaScript
🎤 Selecting male voice...
✅ Voice selected: Microsoft Guy Online (Natural) - Male
⌨️ Entering test text...
✅ Text entered via JavaScript
🔊 Clicking Quick Play...
✅ Quick Play clicked via JavaScript
🎯 DOWNLOAD - Correct Playwright pattern...
✅ Download button visible!
📄 Download captured: audio.mp3
💾 Saving to: /path/to/debug_download/working_test_download.mp3
🎉 DOWNLOAD SUCCESS!
   📍 Location: /path/to/debug_download/working_test_download.mp3
   📊 Size: 45,234 bytes
   ✅ File looks valid!
   ✅ Download issue SOLVED!
```

## ❌ Troubleshooting

### "ModuleNotFoundError: No module named 'playwright'"
```bash
pip install playwright
playwright install chromium
```

### Downloads go to wrong folder
- ✅ **This is FIXED** - The script uses `download.save_as()` which gives exact path control
- The file will be exactly where specified in the script

### Script hangs or times out
- Use `--no-headless` to see what's happening
- Check your internet connection
- Website might be down or changed structure

### Permission errors
```bash
chmod +x playwright_tts_download.py
chmod +x tts_download_production.py
```

## 💡 Key Differences from Original

| Issue | Your Original | Fixed Version |
|-------|---------------|---------------|
| JavaScript syntax | ❌ Broken | ✅ Fixed |
| String closing | ❌ Unclosed | ✅ Properly closed |
| Download method | ✅ Correct | ✅ Correct |
| Error handling | ❌ Basic | ✅ Robust |
| Progress tracking | ❌ None | ✅ Full stats |

## 🎉 Success Criteria

The download is successful when you see:
- ✅ "🎉 DOWNLOAD SUCCESS!"
- ✅ File size > 1,000 bytes
- ✅ File exists in output directory
- ✅ File plays in media player

## 📞 Support

If you still have issues:
1. Run with `--no-headless` to see the browser
2. Check the console output for specific errors
3. Verify `playwright install chromium` completed successfully
4. Try the simple test script first: `python3 playwright_tts_download.py`

---

**Bottom Line**: The syntax errors are fixed. The download method is correct. It should work now! 🚀
