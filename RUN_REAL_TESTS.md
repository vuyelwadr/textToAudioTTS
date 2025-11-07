# 🧪 REAL PERFORMANCE TEST GUIDE - Run This On Your Machine!

**I cannot run real tests from my environment** (network restrictions).

**YOU need to run these tests** and report back the results!

---

## 🎯 Quick Test (5 minutes)

Run this **RIGHT NOW** to get basic performance data:

```bash
# Test 1: Single concurrent download
python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 1 \
    --test-one

# Test 2: 3 concurrent downloads
python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 3 \
    --chunk-size 3000

# Test 3: 5 concurrent downloads
python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 5 \
    --chunk-size 3000
```

**Time each test and tell me:**
- Did it succeed?
- How long did it take?
- Any failures?

---

## 🔬 Full Performance Test (30 minutes)

Run the automated test suite:

```bash
python3 test_real_concurrent.py
```

This will:
- Test at concurrency levels: 1, 3, 5, 10, 20
- Run 5-10 chunks per test
- Measure success rate, throughput, and timing
- Save results to JSON file

**Then send me the output!**

---

## 📊 Manual Test Protocol

If automated test fails, run these manually:

### Test 1: Baseline (1 concurrent)
```bash
time python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 1 \
    --chunk-size 3000 \
    --output-dir test_1_concurrent

# Record:
# - Total time: _____ seconds
# - Successful chunks: _____ / _____
# - Failures: _____
```

### Test 2: Low Concurrency (3 concurrent)
```bash
time python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 3 \
    --chunk-size 3000 \
    --output-dir test_3_concurrent

# Record:
# - Total time: _____ seconds
# - Successful chunks: _____ / _____
# - Failures: _____
# - Speedup vs Test 1: _____x
```

### Test 3: Medium Concurrency (5 concurrent)
```bash
time python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 5 \
    --chunk-size 3000 \
    --output-dir test_5_concurrent

# Record:
# - Total time: _____ seconds
# - Successful chunks: _____ / _____
# - Failures: _____
# - Speedup vs Test 1: _____x
```

### Test 4: High Concurrency (10 concurrent)
```bash
time python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 10 \
    --chunk-size 3000 \
    --output-dir test_10_concurrent

# Record:
# - Total time: _____ seconds
# - Successful chunks: _____ / _____
# - Failures: _____
# - Speedup vs Test 1: _____x
```

### Test 5: Maximum Concurrency (20 concurrent)
```bash
time python3 tts_parallel_fixed.py test_tts_small.txt \
    --concurrent 20 \
    --chunk-size 3000 \
    --output-dir test_20_concurrent

# Record:
# - Total time: _____ seconds
# - Successful chunks: _____ / _____
# - Failures: _____
# - Speedup vs Test 1: _____x
# - Any rate limiting? _____
```

---

## 📝 Results Template

Fill this in and send back to me:

```
=== TTS PARALLEL DOWNLOAD PERFORMANCE TEST RESULTS ===

System Info:
- OS: macOS / Linux / Windows
- CPU: ___________________
- RAM: ___________________
- Internet: _______________ Mbps

Test Results:

| Concurrent | Chunks | Success | Time(s) | Time(min) | Chunks/min | Notes |
|-----------|--------|---------|---------|-----------|------------|-------|
| 1         |        |    /    |         |           |            |       |
| 3         |        |    /    |         |           |            |       |
| 5         |        |    /    |         |           |            |       |
| 10        |        |    /    |         |           |            |       |
| 20        |        |    /    |         |           |            |       |

Observations:
- Optimal concurrency level: _____
- Maximum throughput: _____ chunks/minute
- Success rate dropped at: _____ concurrent
- Rate limiting observed: Yes / No
- Average chunk processing time: _____ seconds

Failures:
- Total failures: _____
- Most common error: _____________________________
- Failed at concurrency: _____

Recommendation:
Best setting for production: --concurrent _____ --chunk-size _____

```

---

## 🎯 What I Need From You

**Minimum** (5 minutes):
Run ONE test and tell me:
```bash
python3 tts_parallel_fixed.py test_tts_small.txt --concurrent 5
```
- How long?
- How many succeeded?
- Any errors?

**Ideal** (30 minutes):
Run the full test suite:
```bash
python3 test_real_concurrent.py > test_results.txt 2>&1
```
Then send me `test_results.txt`

---

## 🔍 What to Look For

### Success Indicators:
- ✅ 90%+ success rate
- ✅ Consistent download times
- ✅ No rate limiting errors
- ✅ Faster wall-clock time with more concurrency

### Warning Signs:
- ⚠️ <80% success rate → reduce concurrency
- ⚠️ "Rate limit" errors → reduce concurrency
- ⚠️ Slower with more concurrency → hitting bottleneck
- ⚠️ "Download button disabled" → chunks too large or slow server

### Failure Indicators:
- ❌ Many timeouts → increase PAGE_TIMEOUT
- ❌ "Button still disabled" → reduce CHUNK_SIZE
- ❌ Inconsistent failures → website rate limiting

---

## 💡 Expected Performance (Typical)

Based on typical TTS automation:

| Concurrent | Expected Time (10 chunks) | Chunks/min | Success Rate |
|-----------|---------------------------|-----------|--------------|
| 1         | ~4-6 minutes              | 2-3       | 95-100%      |
| 3         | ~2-3 minutes              | 4-6       | 90-100%      |
| 5         | ~1.5-2 minutes            | 5-8       | 85-95%       |
| 10        | ~1-1.5 minutes            | 7-10      | 70-90%       |
| 20        | ~1-2 minutes              | 5-10      | 50-80%       |

**Typical optimal: 5-10 concurrent downloads**

But this depends on:
- Website rate limiting
- Your internet speed
- Server response time
- Chunk size

---

## 🚀 Quick Start

```bash
# 1. Create test file
cat > test.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
This is a test of the text to speech system.
Multiple sentences help test chunking behavior.
Each chunk should download successfully.
EOF

# 2. Test increasing concurrency
for concurrent in 1 3 5 10; do
    echo "=== Testing $concurrent concurrent ==="
    time python3 tts_parallel_fixed.py test.txt \
        --concurrent $concurrent \
        --chunk-size 3000 \
        --output-dir test_${concurrent}_concurrent
    echo ""
done

# 3. Check results
ls -lh test_*_concurrent/
```

---

## 📞 Report Back Format

Send me this:

```
Tested tts_parallel_fixed.py on macOS/Linux/Windows

Quick test (5 chunks, concurrent 5):
- Time: 2m 15s
- Success: 5/5
- Speed: 2.2 chunks/min
✅ WORKS!

Full test results:
- Best concurrency: 5
- Max throughput: 8.3 chunks/minute
- Optimal setting: --concurrent 5 --chunk-size 3000
- Success rate: 95%

[paste test output here]
```

---

## ❌ I Cannot Test This Myself

I'm in a sandboxed environment with:
- No access to external websites
- Network proxy restrictions
- No display server for non-headless mode

**You MUST run these tests on your machine and report results back!**

---

## ✅ After Testing

Once you have results, tell me:

1. **What worked:**
   - Best concurrency level?
   - Success rate?
   - Chunks per minute?

2. **What failed:**
   - At what concurrency?
   - What errors?
   - Rate limiting?

3. **What you need:**
   - Different chunk size?
   - Retry logic?
   - Better error handling?

Then I can optimize the script based on **REAL** performance data!

---

**RUN THE TESTS NOW AND SEND ME RESULTS!** 🚀
