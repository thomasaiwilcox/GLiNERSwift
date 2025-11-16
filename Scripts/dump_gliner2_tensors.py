#!/usr/bin/env python3
"""Dump intermediate GLiNER2 tensors for a specific fixture test case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from gliner2 import GLiNER2

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "Tests" / "GLiNERSwiftTests" / "Fixtures" / "python_outputs.json"


def load_case(case_id: str) -> Dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for case in data.get("test_cases", []):
        if case.get("id") == case_id:
            return case
    available = ", ".join(case.get("id", "?") for case in data.get("test_cases", []))
    raise SystemExit(f"Test case '{case_id}' not found. Available cases: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="case_id", default="simple_person", help="Fixture test case id to run")
    parser.add_argument(
        "--model",
        default="fastino/gliner2-base-v1",
        help="GLiNER2 model identifier to load via Hugging Face",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "Scripts" / "debug" / "gliner2_tensor_dump.json"),
        help="Path to write the captured tensors as JSON",
    )
    args = parser.parse_args()

    case = load_case(args.case_id)
    text = case["text"].strip()
    labels = case["labels"]
    threshold = case.get("threshold", 0.5)

    print(f"Loading GLiNER2 model '{args.model}'…")
    extractor = GLiNER2.from_pretrained(args.model)
    extractor.to(torch.device("cpu"))
    extractor.eval()

    captured: Dict[str, Any] = {
        "case_id": case["id"],
        "text": text,
        "labels": labels,
        "threshold": threshold,
    }

    original_predict = GLiNER2._predict_spans

    @torch.no_grad()
    def hooked_predict(self: GLiNER2, embeddings, field_names, span_info):  # type: ignore[override]
        schema_emb = torch.stack(embeddings)
        count_logits = self.count_pred(schema_emb[0].unsqueeze(0))
        pred_count = int(count_logits.argmax(dim=1).item())

        captured["field_names"] = field_names
        captured["schema_embeddings"] = schema_emb.detach().cpu().tolist()
        captured["count_logits"] = count_logits.detach().cpu().tolist()
        captured["predicted_count"] = pred_count
        captured["span_rep_shape"] = list(span_info["span_rep"].shape)
        captured["span_rep"] = span_info["span_rep"].detach().cpu().tolist()
        captured["span_mask"] = span_info["span_mask"].detach().cpu().tolist()
        captured["span_indices"] = span_info["spans_idx"].detach().cpu().tolist()

        if pred_count <= 0:
            captured["structure_embeddings"] = []
            captured["span_scores"] = []
            return 0, None

        struct_proj = self.count_embed(schema_emb[1:], pred_count)
        span_scores = torch.sigmoid(torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj))

        captured["structure_embeddings"] = struct_proj.detach().cpu().tolist()
        captured["span_scores"] = span_scores.detach().cpu().tolist()

        return pred_count, span_scores

    GLiNER2._predict_spans = hooked_predict  # type: ignore[assignment]

    try:
        print(f"Running extraction for case '{case['id']}'…")
        raw_result = extractor.extract_entities(
            text,
            labels,
            threshold=threshold,
            format_results=False,
            include_confidence=True,
        )
        captured["raw_result"] = raw_result
    finally:
        GLiNER2._predict_spans = original_predict  # type: ignore[assignment]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(captured, handle, indent=2)
    print(f"✓ Wrote tensor dump to {output_path}")


if __name__ == "__main__":
    main()
