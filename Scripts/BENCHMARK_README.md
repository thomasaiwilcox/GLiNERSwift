# GLiNER2 Benchmark Comparison Guide

This directory contains benchmarking tools to compare Python GLiNER2 vs Swift GLiNERSwift performance on the same datasets.

## Quick Start

### Run Both Benchmarks and Compare

```bash
cd Scripts
python3 compare_benchmarks.py
```

This will:
1. Run Python GLiNER2 benchmark
2. Run Swift GLiNERSwift benchmark
3. Display a comparison table with speedup metrics

### Example Output

```
📊 BENCHMARK COMPARISON: Python GLiNER2 vs Swift GLiNERSwift
================================================================================

Metric                    Python               Swift                Speedup             
-------------------------------------------------------------------------------------
Samples                   51                   51                   -                   
Total Requests            255                  255                  -                   
Avg Latency (ms)          45.23                12.34                3.67x faster        
Median Latency (ms)       43.10                11.89                3.63x faster        
P95 Latency (ms)          52.87                15.21                3.48x faster        
Min Latency (ms)          38.45                9.12                 4.22x faster        
Max Latency (ms)          61.23                18.45                3.32x faster        
Chars/Second              28453.21             104321.45            3.67x faster        

🏆 WINNER: Swift is 3.67x faster on average! 🚀
```

## Individual Benchmarks

### Python GLiNER2 Benchmark

```bash
# Basic usage (uses default fixtures)
python3 benchmarks.py

# Custom model and device
python3 benchmarks.py --model numind/NuNER-v1.0 --device cpu

# Custom iterations and warmup
python3 benchmarks.py --iterations 10 --warmup 2

# Save results to JSON
python3 benchmarks.py --output results/python_benchmark.json

# Use custom fixtures
python3 benchmarks.py --fixtures ../Sources/Benchmarks/Resources/Fixtures/benchmark_samples_large.jsonl
```

**Options:**
- `--fixtures PATH` - Path to JSONL fixtures (default: benchmark_samples.jsonl)
- `--model NAME` - HuggingFace model name (default: numind/NuNER-v1.0)
- `--iterations N` - Iterations per sample (default: 5)
- `--warmup N` - Warmup passes (default: 1)
- `--device DEVICE` - cpu or cuda (default: cpu)
- `--output PATH` - Save JSON results

### Swift GLiNERSwift Benchmark

```bash
# Build and run (from repo root)
swift run -c release gliner-benchmarks

# With custom options
swift run -c release gliner-benchmarks \
  --fixtures Sources/Benchmarks/Resources/Fixtures/benchmark_samples.jsonl \
  --iterations 10 \
  --warmup 2 \
  --encoder-compute-units cpu-gpu \
  --output results/swift_benchmark.json

# Single text test
swift run -c release gliner-benchmarks \
  --text "John Smith works at Apple Inc. in Cupertino." \
  --labels person,organization,location \
  --iterations 10
```

**Options:**
- `--manifest PATH` - CoreML manifest (default: CoreMLArtifacts/export_manifest.json)
- `--fixtures PATH` - JSONL fixtures
- `--iterations N` - Iterations per sample (default: 5)
- `--warmup N` - Warmup passes (default: 1)
- `--mode MODE` - latency or throughput (default: latency)
- `--encoder-compute-units UNITS` - cpu, cpu-gpu, or all (default: cpu-gpu)
- `--output PATH` - Save JSON results
- `--print-entities` - Display extracted entities

## Comparison Script Options

```bash
# Skip Python benchmark (use cached results)
python3 compare_benchmarks.py --skip-python

# Skip Swift benchmark
python3 compare_benchmarks.py --skip-swift

# Use different Python model
python3 compare_benchmarks.py --python-model "your-org/your-model"

# Run on GPU (Python only, Swift uses CoreML)
python3 compare_benchmarks.py --device cuda

# More iterations for better accuracy
python3 compare_benchmarks.py --iterations 10 --warmup 3

# Custom manifest for Swift
python3 compare_benchmarks.py --manifest ../CoreMLArtifacts/export_manifest.json
```

## Fixture Format

Fixtures are JSONL files with one JSON object per line:

```json
{
  "id": "benchmark_01",
  "text": "John Smith met with OpenAI in San Francisco to discuss GPT-4.",
  "labels": ["person", "organization", "location", "product"],
  "threshold": 0.35
}
```

**Fields:**
- `id` - Unique identifier for the sample
- `text` - Input text to process
- `labels` - Entity types to extract
- `threshold` - Optional confidence threshold (default: 0.3)

## Creating Custom Fixtures

```python
import json

samples = [
    {
        "id": "test_01",
        "text": "Your test text here",
        "labels": ["person", "location"],
        "threshold": 0.3
    },
    # ... more samples
]

with open("my_fixtures.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
```

Then run:
```bash
python3 compare_benchmarks.py --fixtures my_fixtures.jsonl
```

## Prerequisites

### Python Requirements

```bash
# Install GLiNER2
pip install gliner2

# Or for comparison with GLiNER v1
pip install gliner
```

### Swift Requirements

```bash
# Make sure you have Swift toolchain
swift --version

# CoreML models should be converted
cd Scripts
python3 convert_to_coreml.py
```

## Understanding the Results

### Latency Metrics

- **Average Latency** - Mean processing time per request
- **Median Latency** - 50th percentile (less affected by outliers)
- **P95 Latency** - 95th percentile (worst-case typical performance)
- **Min/Max Latency** - Best and worst single request times

### Throughput Metrics

- **Chars/Second** - Characters processed per second
- Higher is better for throughput

### Speedup Calculation

Speedup is calculated as:
- For latency: `Python Latency / Swift Latency`
- For throughput: `Swift Throughput / Python Throughput`

Values > 1.0 mean Swift is faster.

## Troubleshooting

### Python benchmark fails with "gliner2 package missing"

```bash
pip install gliner2
```

### Swift benchmark fails with "Manifest not found"

```bash
# Convert models first
cd Scripts
python3 convert_to_coreml.py

# Or specify manifest location
python3 compare_benchmarks.py --manifest /path/to/export_manifest.json
```

### Python runs on CPU despite --device cuda

Check PyTorch installation:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA-enabled PyTorch.

### Swift uses all CPU cores

This is expected - CoreML automatically parallelizes on CPU. Use `--encoder-compute-units all` to use ANE (Apple Neural Engine) on supported hardware.

## Advanced Usage

### Batch Testing Different Models

```bash
#!/bin/bash
for model in numind/NuNER-v1.0 urchade/gliner_small-v2.1; do
    echo "Testing $model"
    python3 compare_benchmarks.py \
        --python-model "$model" \
        --iterations 10 \
        --output "results/${model//\//_}.json"
done
```

### Memory Profiling

```bash
# Python
python3 -m memory_profiler benchmarks.py

# Swift
swift run -c release gliner-benchmarks 2>&1 | grep "Memory"
```

### CPU vs GPU vs ANE Comparison

```bash
# Run Swift with different compute units
for units in cpu cpu-gpu all; do
    swift run -c release gliner-benchmarks \
        --encoder-compute-units $units \
        --output "results/swift_$units.json"
done
```

## Contributing Benchmarks

To add new benchmark scenarios:

1. Create fixture file in `Sources/Benchmarks/Resources/Fixtures/`
2. Follow the JSONL format above
3. Add diverse text samples with varying lengths and entity densities
4. Include representative entity types
5. Test with both Python and Swift benchmarks

## Performance Tips

### For Python:
- Use `--device cuda` if you have a GPU
- Increase batch size for throughput testing
- Use compiled models when available

### For Swift:
- Use `--encoder-compute-units all` on M-series Macs for ANE acceleration
- Release builds (`-c release`) are ~10x faster than debug
- Warmup is important for stable measurements

## License

Same as parent project.
