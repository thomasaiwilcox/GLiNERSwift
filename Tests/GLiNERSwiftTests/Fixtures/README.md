# Test Fixtures

This directory contains test fixtures generated from the Python GLiNER2 implementation.

## Generating Fixtures

Run the Python script to generate test fixtures:

```bash
cd ../../Scripts
pip install -r requirements.txt
python generate_test_fixtures.py
```

This will create `python_outputs.json` in this directory.

## Fixture Format

The JSON file contains:
- **metadata**: Model information and tolerance thresholds for comparisons
- **test_cases**: Array of test cases with:
  - Input text and labels
  - Tokenizer outputs (token IDs, attention masks)
  - Encoder outputs (hidden states)
  - Expected entity predictions

## Usage in Tests

The test suite loads these fixtures and compares Swift implementation outputs against Python outputs to ensure parity.
