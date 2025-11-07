#!/bin/bash
# TTS CONCURRENCY TEST SCRIPT
# Run this on your Mac to find the optimal concurrency level

echo "=============================================================="
echo "TTS CONCURRENCY TEST - Finding Optimal Settings"
echo "=============================================================="
echo ""

# Create test file with 10 small chunks
cat > /tmp/tts_test.txt << 'EOF'
First test chunk. The quick brown fox jumps over the lazy dog.

Second test chunk. Testing concurrent downloads with different settings.

Third test chunk. Measuring performance and success rates carefully.

Fourth test chunk. Finding the sweet spot for parallel processing.

Fifth test chunk. Balancing speed with reliability is important.

Sixth test chunk. Too much concurrency causes failures and slowdowns.

Seventh test chunk. Too little concurrency wastes time unnecessarily.

Eighth test chunk. The optimal setting depends on your network.

Ninth test chunk. And the server's rate limiting policies matter.

Tenth test chunk. These tests will find your best configuration.
EOF

echo "✅ Created test file with 10 chunks"
echo ""

# Test function
run_test() {
    local concurrent=$1
    local test_name=$2

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 TEST: $test_name (concurrent=$concurrent)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Clean output directory
    rm -rf /tmp/tts_test_output_${concurrent}

    # Run test
    START=$(date +%s)
    python3 parallel_tts_downloader.py \
        /tmp/tts_test.txt \
        --output-dir /tmp/tts_test_output_${concurrent} \
        --concurrent ${concurrent} \
        --chunk-size 200 \
        2>&1 | grep -E '(SUCCESS|Failed|Successful|Time:|concurrent)'

    END=$(date +%s)
    DURATION=$((END - START))

    # Count successes
    SUCCESS_COUNT=$(ls /tmp/tts_test_output_${concurrent}/segment_*.mp3 2>/dev/null | wc -l)

    echo ""
    echo "📊 Results for concurrent=${concurrent}:"
    echo "   ✅ Successful: ${SUCCESS_COUNT}/10"
    echo "   ⏱️  Total time: ${DURATION}s"
    echo "   📈 Rate: $(echo "scale=2; ${SUCCESS_COUNT}/${DURATION}" | bc) chunks/sec"
    echo ""

    # Store results
    echo "${concurrent},${SUCCESS_COUNT},${DURATION}" >> /tmp/tts_concurrency_results.csv
}

# Initialize results file
echo "Concurrency,Success,Duration" > /tmp/tts_concurrency_results.csv

# Run tests
echo "Starting concurrency tests..."
echo ""

run_test 1 "Sequential (baseline)"
sleep 5

run_test 2 "Low concurrent"
sleep 5

run_test 3 "Medium concurrent"
sleep 5

run_test 5 "High concurrent"
sleep 5

run_test 10 "Very high concurrent"
sleep 5

run_test 20 "Extreme concurrent"

# Summary
echo ""
echo "=============================================================="
echo "📊 CONCURRENCY TEST SUMMARY"
echo "=============================================================="
echo ""
cat /tmp/tts_concurrency_results.csv | column -t -s ','
echo ""

# Find best setting
echo "🎯 RECOMMENDATIONS:"
echo ""

# Best success rate
BEST_SUCCESS=$(tail -n +2 /tmp/tts_concurrency_results.csv | sort -t',' -k2 -nr | head -1)
BEST_CONCURRENT=$(echo $BEST_SUCCESS | cut -d',' -f1)
BEST_COUNT=$(echo $BEST_SUCCESS | cut -d',' -f2)
echo "   Best success rate: concurrent=${BEST_CONCURRENT} (${BEST_COUNT}/10 succeeded)"

# Best speed (among those with 80%+ success)
BEST_SPEED=$(tail -n +2 /tmp/tts_concurrency_results.csv | awk -F',' '$2 >= 8' | sort -t',' -k3 -n | head -1)
if [ ! -z "$BEST_SPEED" ]; then
    SPEED_CONCURRENT=$(echo $BEST_SPEED | cut -d',' -f1)
    SPEED_TIME=$(echo $BEST_SPEED | cut -d',' -f3)
    echo "   Fastest (80%+ success): concurrent=${SPEED_CONCURRENT} (${SPEED_TIME}s)"
fi

echo ""
echo "💡 Use these results to set your optimal --concurrent value"
echo "=============================================================="
