#!/bin/bash
# Quick benchmark examples for GLiNER2 Python vs Swift comparison

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPTS_DIR")"

echo "🧪 GLiNER2 Benchmark Examples"
echo "=============================="
echo ""

show_help() {
    echo "Usage: ./quick_benchmark.sh [option]"
    echo ""
    echo "Options:"
    echo "  1, quick        Run quick 3-sample test (default)"
    echo "  2, standard     Run standard 51-sample benchmark"
    echo "  3, python       Run Python benchmark only"
    echo "  4, swift        Run Swift benchmark only"
    echo "  5, compare      Run full comparison"
    echo "  help            Show this help message"
    echo ""
}

quick_test() {
    echo "📝 Running quick test with 3 samples..."
    echo ""
    cd "$SCRIPTS_DIR"
    python3 compare_benchmarks.py \
        --fixtures test_fixtures.jsonl \
        --iterations 2 \
        --warmup 1
}

standard_benchmark() {
    echo "📊 Running standard benchmark with all fixtures..."
    echo ""
    cd "$SCRIPTS_DIR"
    python3 compare_benchmarks.py \
        --iterations 5 \
        --warmup 1
}

python_only() {
    echo "🐍 Running Python GLiNER2 benchmark only..."
    echo ""
    cd "$SCRIPTS_DIR"
    python3 benchmarks.py \
        --iterations 5 \
        --warmup 1 \
        --output /tmp/python_benchmark.json
    echo ""
    echo "Results saved to /tmp/python_benchmark.json"
}

swift_only() {
    echo "🚀 Running Swift GLiNERSwift benchmark only..."
    echo ""
    cd "$REPO_ROOT"
    swift run -c release gliner-benchmarks \
        --iterations 5 \
        --warmup 1 \
        --output /tmp/swift_benchmark.json
    echo ""
    echo "Results saved to /tmp/swift_benchmark.json"
}

full_compare() {
    echo "⚡ Running full comparison benchmark..."
    echo ""
    cd "$SCRIPTS_DIR"
    
    # Create results directory
    mkdir -p "$REPO_ROOT/benchmark_results"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$REPO_ROOT/benchmark_results/$TIMESTAMP"
    mkdir -p "$OUTPUT_DIR"
    
    python3 compare_benchmarks.py \
        --iterations 10 \
        --warmup 2 \
        2>&1 | tee "$OUTPUT_DIR/benchmark_log.txt"
    
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "   - benchmark_log.txt"
    echo "   - /tmp/python_gliner_benchmark.json"
    echo "   - /tmp/swift_gliner_benchmark.json"
}

# Parse command
COMMAND="${1:-quick}"

case "$COMMAND" in
    1|quick)
        quick_test
        ;;
    2|standard)
        standard_benchmark
        ;;
    3|python)
        python_only
        ;;
    4|swift)
        swift_only
        ;;
    5|compare)
        full_compare
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown option: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"
