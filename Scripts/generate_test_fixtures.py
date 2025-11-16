#!/usr/bin/env python3
"""
Generate test fixtures for GLiNER2 Swift package validation.

This script runs the Python GLiNER2 implementation on test cases and exports:
- Tokenizer outputs (token IDs, attention masks)
- Encoder outputs (hidden states)
- Final entity predictions

Outputs are saved to JSON for comparison in Swift unit tests.

Usage:
    python generate_test_fixtures.py --model fastino/gliner2-base-v1
    python generate_test_fixtures.py --model /path/to/custom/model --output ./custom_fixtures.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def patch_gliner2_for_fixtures():
    from gliner2 import GLiNER2

    def _find_valid_spans_with_metadata(
        self,
        field_scores,
        threshold,
        record,
        outputs,
        text_len
    ):
        if threshold is None:
            threshold = 0.5

        valid_positions = torch.where(field_scores >= threshold)
        starts, widths = valid_positions

        spans = []
        start_map = outputs.get("start_token_idx_to_text_idx") or []
        end_map = outputs.get("end_token_idx_to_text_idx") or []
        has_char_mapping = bool(start_map) and bool(end_map)
        text_tokens = outputs.get("text_tokens", [])
        text_tokens = text_tokens[-text_len:] if text_len > 0 else text_tokens

        for start, width in zip(starts.tolist(), widths.tolist()):
            end = start + width + 1
            if 0 <= start < text_len and end <= text_len:
                char_start = char_end = None
                if has_char_mapping:
                    try:
                        char_start = start_map[start]
                        char_end = end_map[end - 1]
                    except (IndexError, KeyError):
                        char_start = char_end = None

                if char_start is not None and char_end is not None:
                    text_span = record["text"][char_start:char_end].strip()
                else:
                    span_tokens = text_tokens[start:end]
                    text_span = " ".join(span_tokens).strip()

                confidence = field_scores[start, width].item()
                if text_span:
                    spans.append((
                        text_span,
                        confidence,
                        char_start if char_start is not None else -1,
                        char_end if char_end is not None else -1,
                        start,
                        end
                    ))

        return spans

    def _format_span_list_with_metadata(self, spans):
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        selected = []

        for text, confidence, char_start, char_end, token_start, token_end in sorted_spans:
            overlap = False
            for existing in selected:
                existing_token_start = existing["token_start"]
                existing_token_end = existing["token_end"]
                if not (token_end <= existing_token_start or token_start >= existing_token_end):
                    overlap = True
                    break
            if overlap:
                continue
            selected.append({
                "text": text,
                "score": float(confidence),
                "start": char_start,
                "end": char_end,
                "token_start": token_start,
                "token_end": token_end
            })

        return selected

    GLiNER2._find_valid_spans = _find_valid_spans_with_metadata
    GLiNER2._format_span_list = _format_span_list_with_metadata


# Test cases for validation
TEST_CASES = [
    {
        "id": "simple_person",
        "text": "John Smith works at Apple Inc.",
        "labels": ["person", "organization"],
        "threshold": 0.5
    },
    {
        "id": "multi_entity",
        "text": "Barack Obama was the 44th President of the United States. He was born in Honolulu, Hawaii.",
        "labels": ["person", "location", "position"],
        "threshold": 0.4
    },
    {
        "id": "technical",
        "text": "The API endpoint returns JSON data with a 200 status code.",
        "labels": ["technology", "data format", "status code"],
        "threshold": 0.3
    },
    {
        "id": "empty",
        "text": "This sentence has no relevant entities.",
        "labels": ["person", "organization"],
        "threshold": 0.5
    },
    {
        "id": "long_text",
        "text": " ".join([
            "Artificial intelligence research at Stanford University has made significant progress.",
            "Dr. Emily Chen leads the team working on natural language processing.",
            "The project is funded by the National Science Foundation and private donors.",
            "Results will be published in Nature journal next month."
        ]),
        "labels": ["person", "organization", "location", "publication"],
        "threshold": 0.4
    }
]


def load_gliner_model(model_name_or_path):
    """Load GLiNER2 model."""
    try:
        from gliner2 import GLiNER2
        patch_gliner2_for_fixtures()
        print(f"Loading GLiNER2 model: {model_name_or_path}")
        model = GLiNER2.from_pretrained(model_name_or_path)
        model.eval()
        return model
    except ImportError:
        print("Error: gliner2 package not found. Install with: pip install gliner2")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def process_test_case(model, test_case):
    """Process a single test case and extract all intermediate outputs."""
    text = test_case["text"]
    labels = test_case["labels"]
    threshold = test_case["threshold"]
    
    print(f"  Processing: {test_case['id']}")
    
    # Get tokenizer
    tokenizer = None
    if hasattr(model, 'processor') and hasattr(model.processor, 'tokenizer'):
        tokenizer = model.processor.tokenizer
    elif hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    elif hasattr(model, 'data_processor'):
        tokenizer = model.data_processor.transformer_tokenizer

    if tokenizer is None:
        raise AttributeError("Cannot find tokenizer in GLiNER2 model")
    
    # Tokenize text
    tokenized = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Get encoder outputs
    with torch.no_grad():
        # Try to get encoder directly
        if hasattr(model, 'token_rep_layer'):
            encoder_output = model.token_rep_layer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        elif hasattr(model, 'encoder'):
            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            # Fallback: get hidden states from full forward pass
            encoder_output = None
        
        # Extract hidden states
        if encoder_output is not None:
            if hasattr(encoder_output, 'last_hidden_state'):
                hidden_states = encoder_output.last_hidden_state
            elif isinstance(encoder_output, tuple):
                hidden_states = encoder_output[0]
            else:
                hidden_states = encoder_output
        else:
            hidden_states = None
        
    # Get entity predictions using full model (raw, unformatted)
    extraction = model.extract_entities(text, labels, threshold=threshold, format_results=False)
    entities = []
    entity_results = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
    if isinstance(entity_results, list) and entity_results:
        entity_results = entity_results[0]

    for label in labels:
        spans = entity_results.get(label, []) if isinstance(entity_results, dict) else []
        if not isinstance(spans, list):
            continue
        for span in spans:
            if isinstance(span, dict):
                score = float(span.get("score", 0.0))
                start = int(span.get("start", -1))
                end = int(span.get("end", -1))
                text_value = span.get("text", "")
            else:
                score = 0.0
                start = -1
                end = -1
                text_value = str(span)
            entities.append({
                "text": text_value,
                "label": label,
                "score": score,
                "start": start,
                "end": end
            })
    
    # Convert to serializable format
    fixture = {
        "id": test_case["id"],
        "text": text,
        "labels": labels,
        "threshold": threshold,
        "tokenizer_output": {
            "input_ids": input_ids[0].tolist(),
            "attention_mask": attention_mask[0].tolist(),
            "tokens": tokenizer.convert_ids_to_tokens(input_ids[0])
        },
        "encoder_output": {
            "hidden_states": hidden_states[0].cpu().numpy().tolist() if hidden_states is not None else None,
            "shape": list(hidden_states.shape) if hidden_states is not None else None
        },
        "entities": [
            {
                "text": e["text"],
                "label": e["label"],
                "score": float(e["score"]),
                "start": e["start"],
                "end": e["end"]
            }
            for e in entities
        ]
    }
    
    return fixture


def generate_fixtures(model, test_cases, output_path):
    """Generate all test fixtures."""
    print(f"\nGenerating {len(test_cases)} test fixtures...")
    
    fixtures = {
        "metadata": {
            "model": model.config.model_name if hasattr(model.config, 'model_name') else "unknown",
            "tolerance": {
                "token_ids": 0,  # Exact match
                "attention_mask": 0,  # Exact match
                "hidden_states": 1e-4,  # Floating point tolerance
                "entity_scores": 0.01  # Small tolerance for scores
            },
            "generator_version": "1.0.0"
        },
        "test_cases": []
    }
    
    for test_case in test_cases:
        try:
            fixture = process_test_case(model, test_case)
            fixtures["test_cases"].append(fixture)
            print(f"    ✓ {test_case['id']}: {len(fixture['entities'])} entities found")
        except Exception as e:
            print(f"    ✗ {test_case['id']}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving fixtures to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(fixtures, f, indent=2)
    
    print(f"✓ Saved {len(fixtures['test_cases'])} test fixtures")
    
    # Print summary
    print("\nFixture Summary:")
    print(f"  Total test cases: {len(fixtures['test_cases'])}")
    total_entities = sum(len(tc['entities']) for tc in fixtures['test_cases'])
    print(f"  Total entities: {total_entities}")
    print(f"  Tolerance for hidden states: {fixtures['metadata']['tolerance']['hidden_states']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test fixtures for GLiNER2 Swift package",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="fastino/gliner2-base-v1",
        help="HuggingFace model name or local path (default: fastino/gliner2-base-v1)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="../Tests/GLiNERSwiftTests/Fixtures/python_outputs.json",
        help="Output JSON file path (default: ../Tests/GLiNERSwiftTests/Fixtures/python_outputs.json)"
    )
    
    parser.add_argument(
        "--additional-tests",
        type=str,
        help="Path to JSON file with additional test cases"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GLiNER2 Test Fixture Generator")
    print("=" * 70)
    
    # Load model
    model = load_gliner_model(args.model)
    
    # Load test cases
    test_cases = TEST_CASES.copy()
    
    if args.additional_tests:
        print(f"Loading additional tests from: {args.additional_tests}")
        with open(args.additional_tests) as f:
            additional = json.load(f)
            test_cases.extend(additional)
    
    # Generate fixtures
    generate_fixtures(model, test_cases, args.output)
    
    print("\n" + "=" * 70)
    print("✓ Fixture generation complete!")
    print("=" * 70)
    print(f"Output: {args.output}")
    print("\nNext steps:")
    print("1. Review generated fixtures")
    print("2. Run Swift tests: swift test")
    print("3. Verify parity between Python and Swift implementations")


if __name__ == "__main__":
    main()
