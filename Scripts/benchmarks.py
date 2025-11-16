#!/usr/bin/env python3
"""Run GLiNER2 Python benchmarks that mirror the Swift CLI fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_FIXTURES = Path(__file__).resolve().parents[1] / "Sources" / "Benchmarks" / "Resources" / "Fixtures" / "benchmark_samples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Python GLiNER2 model using shared fixtures")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="Path to the JSONL fixtures file")
    parser.add_argument("--model", type=str, default="numind/NuNER-v1.0", help="GLiNER2 model name/path (default: numind/NuNER-v1.0)")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations per sample (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup passes per sample (default: 1)")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON summary")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on: cpu or cuda (default: cpu)")
    return parser.parse_args()


def load_model(name: str, device: str):
    """Load GLiNER2 model, with fallback to GLiNER v1."""
    # Try GLiNER2 first
    try:
        from gliner2 import GLiNER2
        model = GLiNER2.from_pretrained(name, device=device)
        model.eval()
        return model, "v2"
    except (ImportError, Exception) as e:
        # If GLiNER2 fails (not installed or model incompatible), try GLiNER v1
        try:
            from gliner import GLiNER
            print(f"Warning: GLiNER2 failed ({e.__class__.__name__}), using GLiNER v1 instead")
            model = GLiNER.from_pretrained(name)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            return model, "v1"
        except ImportError as exc:
            raise SystemExit("Neither gliner2 nor gliner package found. Install with `pip install gliner2`.") from exc


def load_fixtures(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Fixtures file not found: {path}")

    fixtures: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fixtures.append(json.loads(line))

    if not fixtures:
        raise SystemExit(f"No fixtures found in {path}")
    return fixtures


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    remainder = rank - lower
    return ordered[lower] * (1 - remainder) + ordered[upper] * remainder


def run_benchmarks(model, model_version: str, fixtures: List[Dict[str, Any]], iterations: int, warmup: int) -> Dict[str, Any]:
    """Run benchmarks with GLiNER2 API (or fallback to v1)."""
    latencies: List[float] = []
    total_characters = 0
    total_entities = 0

    for sample in fixtures:
        text = sample["text"]
        labels = sample["labels"]
        threshold = sample.get("threshold", 0.3)

        # Warmup passes
        for _ in range(max(warmup, 0)):
            if model_version == "v2":
                # GLiNER2 API using Schema builder
                from gliner2 import Schema
                schema = Schema().entities(labels)
                model.extract(text, schema, threshold=threshold)
            else:
                # GLiNER v1 API
                model.predict_entities(text, labels, threshold=threshold)

        # Measured iterations
        for iteration in range(max(iterations, 1)):
            start = time.perf_counter()
            
            if model_version == "v2":
                # GLiNER2 API using Schema builder
                from gliner2 import Schema
                schema = Schema().entities(labels)
                result = model.extract(text, schema, threshold=threshold)
                # Count entities from result
                entity_count = sum(
                    len(entities) if isinstance(entities, list) else 1
                    for entities in result.entities.values()
                )
            else:
                # GLiNER v1 API
                entities = model.predict_entities(text, labels, threshold=threshold)
                entity_count = len(entities)
            
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            total_characters += len(text)
            total_entities += entity_count
            
            print(f"{sample['id']:<20} | iter {iteration + 1:02d} | {elapsed * 1000:.2f} ms | {entity_count} entities")

    avg_latency = statistics.mean(latencies) * 1000 if latencies else 0.0
    median_latency = statistics.median(latencies) * 1000 if latencies else 0.0
    p95_latency = percentile(latencies, 0.95) * 1000
    min_latency = min(latencies, default=0.0) * 1000
    max_latency = max(latencies, default=0.0) * 1000
    total_time = sum(latencies)
    chars_per_second = (total_characters / total_time) if total_time else 0.0

    return {
        "modelVersion": model_version,
        "samples": len(fixtures),
        "iterationsPerSample": max(iterations, 1),
        "totalRequests": len(latencies),
        "averageLatencyMs": avg_latency,
        "medianLatencyMs": median_latency,
        "p95LatencyMs": p95_latency,
        "minLatencyMs": min_latency,
        "maxLatencyMs": max_latency,
        "charactersPerSecond": chars_per_second,
        "totalEntitiesExtracted": total_entities,
    }


def main() -> None:
    args = parse_args()
    fixtures = load_fixtures(args.fixtures)
    
    print(f"Loading model: {args.model} on {args.device}...")
    model, model_version = load_model(args.model, args.device)
    print(f"Using GLiNER {model_version}")
    
    stats = run_benchmarks(model, model_version, fixtures, args.iterations, args.warmup)

    print("\n=== Python GLiNER2 Benchmark Summary ===")
    for key, value in stats.items():
        if key.endswith("Ms"):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
