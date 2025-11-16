# GLiNERSwift Setup Guide

Complete guide to get GLiNERSwift up and running.

## Prerequisites

1. **System Requirements**
   - macOS 13.0+ or iOS 16.0+
   - Xcode 15.0+
   - Swift 5.9+
   - Python 3.9+ (for model conversion)

2. **Python Environment (for model generation)**
   ```bash
   # Recommended: use a virtual environment
   python3 -m venv venv
   source venv/bin/activate
   ```

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/thomasaiwilcox/GLiNERSwift.git
cd GLiNERSwift
```

**Note:** The repository does NOT include the 1.5GB of Core ML models. You must generate them (see Step 2).

### 2. Generate Core ML Models

The models are **not included in git** to keep the repository lightweight. Generate them using the provided script:

```bash
cd Scripts
pip install -r requirements.txt
python convert_to_coreml.py \
    --model urchade/gliner_multi-v2.1 \
    --output ../CoreMLArtifacts \
    --precision float16
```

**What this does:**
- Downloads GLiNER2 model from HuggingFace (~400MB download)
- Converts 5 Core ML components (encoder, span scorer, count predictor, etc.)
- Exports tokenizer files
- Creates `export_manifest.json` with all paths
- Saves to `CoreMLArtifacts/` (~426MB final size)

**Time required:** 5-10 minutes on Apple Silicon Mac

### 3. Build the Package

```bash
cd ..
swift build
```

### 4. Run Tests

```bash
swift test
```

Expected: All 27 tests pass ✅
- Download the GLiNER model from HuggingFace
- Patch the DeBERTa helper functions to avoid Core ML-incompatible ops (`sqrt`, scripted `if`)
- Extract and convert the encoder
- Save `GLiNER2Encoder.mlpackage` to `Sources/GLiNERSwift/Resources/`
- Export tokenizer files

**Expected output:**
```
GLiNER to Core ML (Patched DeBERTa scaling)
======================================================================
[Patch] Rewriting transformers DeBERTa helpers...
...
✓ Core ML model saved: ../Sources/GLiNERSwift/Resources/GLiNER2Encoder.mlpackage
```

## Step 3: Generate Test Fixtures

Generate test data to validate the Swift implementation:

```bash
python generate_test_fixtures.py
```

This creates `Tests/GLiNERSwiftTests/Fixtures/python_outputs.json`.

## Step 4: Build the Package

```bash
cd ..
swift build
```

**Troubleshooting build issues:**

- If you get "module not found" errors, ensure dependencies are resolved:
  ```bash
  swift package resolve
  swift package update
  ```

- If Core ML model is not found, verify:
  ```bash
  ls -la Sources/GLiNERSwift/Resources/
  ```
  You should see `GLiNER2Encoder.mlpackage` and `tokenizer/`

## Step 5: Run Tests

```bash
swift test
```

**Expected output:**
```
Test Suite 'All tests' passed
     Executed 15 tests, with 0 failures
```

## Step 6: Try the Demo App

```bash
cd Examples/GLiNERDemo
swift run
```

Or open in Xcode:
```bash
open Package.swift
```

## Verifying Installation

Quick verification script:

```swift
import GLiNERSwift

Task {
    let model = try await GLiNERModel()
    let entities = try await model.extractEntities(
        from: "Apple Inc. is located in Cupertino.",
        labels: ["organization", "location"]
    )
    print("Found \(entities.count) entities:")
    entities.forEach { print("  - \($0.label): \($0.text)") }
}
```

## Common Issues

### 1. Git LFS Files Not Downloaded

**Symptom:** Build fails with "model not found"

**Solution:**
```bash
git lfs install
git lfs pull
```

### 2. Python Dependencies Conflict

**Symptom:** `pip install` fails

**Solution:**
```bash
python3 -m venv fresh_venv
source fresh_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Core ML Conversion Fails

**Symptom:** "Could not find encoder" or conversion errors

**Solution:**
- Check GLiNER version compatibility
- Try a different model variant:
  ```bash
  python convert_to_coreml.py --model urchade/gliner_multi
  ```

### 4. swift-transformers Not Found

**Symptom:** Build error about Transformers module

**Solution:**
```bash
swift package reset
swift package resolve
swift package update
```

## Next Steps

- Read the [README.md](../README.md) for usage examples
- Check out the [demo app](../Examples/GLiNERDemo/)
- Explore the [API documentation](https://yourusername.github.io/GLiNERSwift)

## Converting Custom Models

If you have a fine-tuned GLiNER2 model:

```bash
python convert_to_coreml.py \
    --model /path/to/your/finetuned/model \
    --output ../Sources/GLiNERSwift/Resources \
    --name CustomGLiNER \
    --max-length 256
```

Then use it in Swift:

```swift
let modelURL = Bundle.main.url(forResource: "CustomGLiNER", withExtension: "mlpackage")!
let tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: nil)!

let model = try await GLiNERModel(
    modelURL: modelURL,
    tokenizerURL: tokenizerURL
)
```

## Support

If you encounter issues:

1. Check [GitHub Issues](https://github.com/yourusername/GLiNERSwift/issues)
2. Review [Discussions](https://github.com/yourusername/GLiNERSwift/discussions)
3. Open a new issue with:
   - OS version
   - Xcode version
   - Full error message
   - Steps to reproduce
