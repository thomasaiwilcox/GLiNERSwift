#!/usr/bin/env python3
"""Compare Python GLiNER2 vs Swift GLiNERSwift performance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

DEFAULT_FIXTURES = Path(__file__).resolve().parents[1] / "Sources" / "Benchmarks" / "Resources" / "Fixtures" / "benchmark_samples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Python GLiNER2 vs Swift GLiNERSwift")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="Path to JSONL fixtures")
    parser.add_argument("--python-model", type=str, default="numind/NuNER-v1.0", help="Python model to benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per sample")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes")
    parser.add_argument("--device", type=str, default="cpu", help="Python device: cpu or cuda")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python benchmark")
    parser.add_argument("--skip-swift", action="store_true", help="Skip Swift benchmark")
    parser.add_argument("--manifest", type=Path, help="Swift CoreML manifest path")
    return parser.parse_args()


def run_python_benchmark(args: argparse.Namespace) -> Dict[str, Any] | None:
    """Run Python benchmark and return stats."""
    if args.skip_python:
        print("⏭️  Skipping Python benchmark")
        return None
    
    print("=" * 80)
    print("🐍 Running Python GLiNER2 Benchmark")
    print("=" * 80)
    
    output_path = Path("/tmp/python_gliner_benchmark.json")
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "benchmarks.py"),
        "--fixtures", str(args.fixtures),
        "--model", args.python_model,
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--device", args.device,
        "--output", str(output_path),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        if output_path.exists():
            with output_path.open("r") as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"❌ Python benchmark failed: {e}")
        return None
    except FileNotFoundError:
        print("❌ Python benchmark script not found")
        return None


def run_swift_benchmark(args: argparse.Namespace) -> Dict[str, Any] | None:
    """Run Swift benchmark and return stats."""
    if args.skip_swift:
        print("⏭️  Skipping Swift benchmark")
        return None
    
    print("\n" + "=" * 80)
    print("🚀 Running Swift GLiNERSwift Benchmark")
    print("=" * 80)
    
    output_path = Path("/tmp/swift_gliner_benchmark.json")
    
    # Find the Swift benchmark executable
    repo_root = Path(__file__).resolve().parents[1]
    
    # Build the Swift package first
    print("Building Swift package...")
    build_cmd = ["swift", "build", "-c", "release"]
    try:
        subprocess.run(build_cmd, cwd=repo_root, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Swift build failed: {e.stderr}")
        return None
    
    # Run the benchmark
    cmd = ["swift", "run", "-c", "release", "gliner-benchmarks"]
    # Convert fixtures path to absolute if it's relative
    fixtures_path = args.fixtures if args.fixtures.is_absolute() else (Path.cwd() / args.fixtures).resolve()
    cmd.extend(["--fixtures", str(fixtures_path)])
    cmd.extend(["--iterations", str(args.iterations)])
    cmd.extend(["--warmup", str(args.warmup)])
    cmd.extend(["--output", str(output_path)])
    
    if args.manifest:
        cmd.extend(["--manifest", str(args.manifest)])
    
    try:
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=False, text=True)
        
        if output_path.exists():
            with output_path.open("r") as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"❌ Swift benchmark failed")
        return None


def format_speedup(swift_val: float, python_val: float) -> str:
    """Format speedup factor."""
    if python_val == 0:
        return "N/A"
    ratio = python_val / swift_val
    if ratio > 1:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1/ratio:.2f}x slower"


def print_comparison(python_stats: Dict[str, Any] | None, swift_stats: Dict[str, Any] | None):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("📊 BENCHMARK COMPARISON: Python GLiNER2 vs Swift GLiNERSwift")
    print("=" * 80)
    
    if not python_stats and not swift_stats:
        print("❌ No benchmarks completed")
        return
    
    metrics = [
        ("Samples", "samples", False),
        ("Total Requests", "totalRequests", False),
        ("Avg Latency (ms)", "averageLatencyMs", True),
        ("Median Latency (ms)", "medianLatencyMs", True),
        ("P95 Latency (ms)", "p95LatencyMs", True),
        ("Min Latency (ms)", "minLatencyMs", True),
        ("Max Latency (ms)", "maxLatencyMs", True),
        ("Chars/Second", "charactersPerSecond", False),
    ]
    
    print(f"\n{'Metric':<25} {'Python':<20} {'Swift':<20} {'Speedup':<20}")
    print("-" * 85)
    
    for label, key, is_latency in metrics:
        python_val = python_stats.get(key) if python_stats else None
        swift_val = swift_stats.get(key) if swift_stats else None
        
        python_str = f"{python_val:.2f}" if python_val is not None else "N/A"
        swift_str = f"{swift_val:.2f}" if swift_val is not None else "N/A"
        
        if is_latency and python_val is not None and swift_val is not None:
            speedup = format_speedup(swift_val, python_val)
        elif key == "charactersPerSecond" and python_val is not None and swift_val is not None:
            speedup = format_speedup(python_val, swift_val)  # Inverted for throughput
        else:
            speedup = "-"
        
        print(f"{label:<25} {python_str:<20} {swift_str:<20} {speedup:<20}")
    
    # Extra info
    print("\n" + "-" * 85)
    if python_stats:
        print(f"Python Model Version: {python_stats.get('modelVersion', 'unknown')}")
    if swift_stats:
        print(f"Swift Mode: {swift_stats.get('mode', 'unknown')}")
    
    # Summary
    if python_stats and swift_stats:
        avg_py = python_stats.get("averageLatencyMs", 0)
        avg_sw = swift_stats.get("averageLatencyMs", 0)
        if avg_py > 0 and avg_sw > 0:
            print(f"\n{'🏆 WINNER:':<25} ", end="")
            if avg_sw < avg_py:
                speedup = avg_py / avg_sw
                print(f"Swift is {speedup:.2f}x faster on average! 🚀")
            else:
                speedup = avg_sw / avg_py
                print(f"Python is {speedup:.2f}x faster on average 🐍")


def main():
    args = parse_args()
    
    if not args.fixtures.exists():
        print(f"❌ Fixtures not found: {args.fixtures}")
        sys.exit(1)
    
    python_stats = run_python_benchmark(args)
    swift_stats = run_swift_benchmark(args)
    
    print_comparison(python_stats, swift_stats)


if __name__ == "__main__":
    main()
