# GLiNERSwift Setup Guide

This guide walks you through setting up GLiNERSwift from source.

## Prerequisites

- macOS 13.0+ or iOS 16.0+
- Xcode 15.0+
- Swift 5.9+
- Python 3.9+ (for model conversion)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tomaarsen/GLiNERSwift.git
cd GLiNERSwift
```

### 2. Generate Core ML Models

The repository does not include the 1.5GB of Core ML model files. Generate them using the provided conversion script:

```bash
cd Scripts
pip install -r requirements.txt
python convert_to_coreml.py \
    --model urchade/gliner_multi-v2.1 \
    --output ../CoreMLArtifacts \
    --precision float16
```

This will:
- Download the GLiNER2 model from HuggingFace (~400MB)
- Convert 5 Core ML components:
  - Encoder (DeBERTa backbone)
  - Span representation model
  - Count predictor
  - Schema projector  
  - Classifier head
- Export tokenizer files
- Create `export_manifest.json`
- Save everything to `CoreMLArtifacts/`

**Expected output size:** ~426MB

**Time:** 5-10 minutes on Apple Silicon Mac

### 3. Build the Swift Package

```bash
cd ..
swift build
```

### 4. Run Tests

```bash
swift test
```

All 27 tests should pass, confirming parity with Python GLiNER2.

### 5. Run Benchmarks

```bash
# Swift benchmark
swift run -c release gliner-benchmarks \
    --fixtures Scripts/test_fixtures.jsonl \
    --iterations 5 \
    --warmup 1

# Compare with Python
cd Scripts
python compare_benchmarks.py \
    --fixtures test_fixtures.jsonl \
    --iterations 5 \
    --python-model urchade/gliner_multi-v2.1
```

## Model Conversion Options

### Using a Different Model

```bash
python convert_to_coreml.py \
    --model your-username/your-gliner2-model \
    --output ../CoreMLArtifacts \
    --precision float16
```

### Full Precision (FP32)

For maximum accuracy at the cost of 2x model size:

```bash
python convert_to_coreml.py \
    --model urchade/gliner_multi-v2.1 \
    --output ../CoreMLArtifacts \
    --precision float32
```

### Custom Sequence Length

```bash
python convert_to_coreml.py \
    --model urchade/gliner_multi-v2.1 \
    --output ../CoreMLArtifacts \
    --max-seq-len 512 \
    --max-schema-tokens 128
```

## Using GLiNERSwift in Your Project

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/tomaarsen/GLiNERSwift.git", branch: "main")
]
```

### Xcode Project

1. File → Add Package Dependencies
2. Enter: `https://github.com/tom/GLiNERSwift.git`
3. Select version/branch
4. Add to your target

**Important:** You must provide Core ML models at runtime. Either:
- Bundle generated models in your app
- Download on first launch
- Use the manifest-based initialization

## Troubleshooting

### "Model files not found"

Make sure you've run the conversion script and the models are in `CoreMLArtifacts/`:

```bash
ls -lh CoreMLArtifacts/*.mlpackage
```

### Python dependencies fail to install

Use a virtual environment:

```bash
cd Scripts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Conversion script fails

Check you have enough disk space (~2GB free) and memory (~4GB free).

For M1/M2 Macs, ensure you're using native Python:

```bash
which python3
# Should be /opt/homebrew/bin/python3 or similar, not /usr/bin/python3
```

### Tests fail

Regenerate test fixtures:

```bash
cd Scripts
python generate_test_fixtures.py
```

## Directory Structure

```
GLiNERSwift/
├── Sources/
│   └── GLiNERSwift/          # Swift implementation
│       ├── API/              # Public API (GLiNER2, Schema)
│       ├── Models/           # Core ML model wrappers
│       ├── Tokenization/     # Custom tokenizer
│       └── Resources/        # Model files (git-ignored, generated)
├── Scripts/
│   ├── convert_to_coreml.py  # Model conversion
│   ├── benchmarks.py         # Python benchmarks
│   ├── compare_benchmarks.py # Comparison tool
│   └── requirements.txt      # Python dependencies
├── Tests/
│   └── GLiNERSwiftTests/     # Test suite
├── CoreMLArtifacts/          # Generated models (git-ignored)
└── Package.swift             # Swift package manifest
```

## CI/CD Setup

For continuous integration, add a setup step to generate models:

```yaml
# .github/workflows/test.yml
- name: Generate Core ML Models
  run: |
    cd Scripts
    pip install -r requirements.txt
    python convert_to_coreml.py \
      --model urchade/gliner_multi-v2.1 \
      --output ../CoreMLArtifacts
```

## Pre-built Models (Optional)

If you prefer not to run the conversion script, you can download pre-built models from [Releases](https://github.com/tomaarsen/GLiNERSwift/releases):

1. Download `CoreMLArtifacts.zip`
2. Extract to project root
3. Verify structure matches above

## License

Apache License 2.0 - Models are derivatives of GLiNER2 (also Apache 2.0)
