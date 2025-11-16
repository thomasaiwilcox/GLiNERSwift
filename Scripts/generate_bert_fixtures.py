#!/usr/bin/env python3
"""
Generate test fixtures using BERT encoder for parity testing.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

print("=" * 60)
print("Generating Test Fixtures with BERT Encoder")
print("=" * 60)

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)
encoder.eval()
print("✓ Model loaded")

# Define test cases
test_cases = [
    {
        "text": "Apple Inc. was founded by Steve Jobs in Cupertino.",
        "labels": ["organization", "person", "location"]
    },
    {
        "text": "The Eiffel Tower in Paris attracts millions of visitors.",
        "labels": ["landmark", "city"]
    },
    {
        "text": "Tesla announced new battery technology yesterday.",
        "labels": ["company", "product", "date"]
    },
]

# Process each test case
test_outputs = []
for i, test in enumerate(test_cases, 1):
    print(f"\nProcessing test case {i}/{len(test_cases)}...")
    text = test["text"]
    labels = test["labels"]
    
    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    # Get encoder outputs
    with torch.no_grad():
        outputs = encoder(**encoded)
        hidden_states = outputs.last_hidden_state[0]  # Remove batch dim
    
    test_output = {
        "text": text,
        "labels": labels,
        "input_ids": encoded["input_ids"][0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "hidden_states": hidden_states.cpu().numpy().tolist()
    }
    
    test_outputs.append(test_output)
    print(f"   Tokenized to {len(encoded['input_ids'][0])} tokens")
    print(f"   Hidden states shape: {hidden_states.shape}")

# Save fixtures
output_dir = Path("../Tests/GLiNERSwiftTests/Fixtures")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "python_outputs.json"

output_data = {
    "model": model_name,
    "hidden_dim": 768,
    "test_cases": test_outputs
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Saved {len(test_outputs)} test fixtures to: {output_file}")
print("\n" + "=" * 60)
print("Test fixture generation complete!")
print("=" * 60)
