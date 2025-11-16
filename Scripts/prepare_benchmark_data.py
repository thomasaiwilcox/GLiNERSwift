#!/usr/bin/env python3
"""Create or extend the shared benchmark fixture set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Sources" / "Benchmarks" / "Resources" / "Fixtures" / "benchmark_samples.jsonl"

SEED_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "business_short",
        "text": "Tim Cook presented the new iPhone lineup to investors in Cupertino.",
        "labels": ["person", "organization", "location"],
        "threshold": 0.45,
    },
    {
        "id": "health_research",
        "text": "Dr. Priya Patel and the Horizon Oncology Center announced a breakthrough melanoma therapy at Johns Hopkins University.",
        "labels": ["person", "organization", "location"],
        "threshold": 0.4,
    },
    {
        "id": "policy_update",
        "text": "The European Commission signed a clean energy partnership with NorthWind Solar and NovaGrid in Brussels.",
        "labels": ["organization", "location", "agreement"],
        "threshold": 0.35,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark fixtures for Swift/Python harnesses")
    parser.add_argument("--input", type=Path, help="Optional JSON or JSONL file with samples to merge")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL path")
    parser.add_argument("--append", action="store_true", help="Append to existing fixtures instead of replacing")
    return parser.parse_args()


def read_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples", [])
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list or contain a 'samples' list")
    return data  # type: ignore[return-value]


def validate(sample: Dict[str, Any]) -> Dict[str, Any]:
    required = {"id", "text", "labels"}
    missing = required - sample.keys()
    if missing:
        raise SystemExit(f"Sample missing keys {missing}: {sample}")
    if not isinstance(sample["labels"], list) or not sample["labels"]:
        raise SystemExit(f"Sample must include at least one label: {sample['id']}")
    return {
        "id": str(sample["id"]),
        "text": str(sample["text"]),
        "labels": [str(label) for label in sample["labels"]],
        "threshold": float(sample.get("threshold", 0.4)),
    }


def write_jsonl(entries: Iterable[Dict[str, Any]], path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        if append and path.exists() and path.stat().st_size > 0:
            handle.write("")
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    samples: List[Dict[str, Any]]
    if args.input:
        samples = read_samples(args.input)
    else:
        samples = SEED_SAMPLES

    validated = [validate(sample) for sample in samples]
    write_jsonl(validated, args.output, args.append)

    print(f"Wrote {len(validated)} benchmark samples to {args.output}")


if __name__ == "__main__":
    main()
