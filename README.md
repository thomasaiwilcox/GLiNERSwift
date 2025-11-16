# GLiNERSwift

A Swift package for on-device zero-shot named entity recognition (NER) using GLiNER2 with Core ML. Full feature parity with Python GLiNER2, including Schema API, text classification, and structured data extraction.

## Features

- 🚀 **On-Device Inference**: Run GLiNER2 entirely on-device with Core ML (CPU+GPU)
- 🎯 **Zero-Shot NER**: Extract entities without training—just provide label prompts
- 🔧 **GLiNER2 Schema API**: Fluent Schema builder for multi-task extraction (entities, classification, structured data)
- 🏷️ **Text Classification**: Multi-label and single-label text classification with sigmoid/softmax
- 📊 **Structured Extraction**: Extract structured data with field-level configuration
- 📦 **Pure Swift**: Native implementation with custom tokenizer, no Python dependencies at runtime
- 🔄 **Custom Models**: Convert your fine-tuned GLiNER2 models to Core ML
- 💻 **Cross-Platform**: macOS 13+ and iOS 16+

## Installation

### Swift Package Manager

Add GLiNERSwift to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/thomasaiwilcox/GLiNERSwift.git", branch: "main")
]
```

### Model Setup

The Core ML models are **not included in this repository** to keep it lightweight. After cloning, you need to generate them:

```bash
cd Scripts
python3 convert_to_coreml.py
```

This will download the GLiNER2 model from HuggingFace and convert it to Core ML format (~1.5GB).

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Quick Start

### Basic Entity Extraction

```swift
import GLiNERSwift

// Initialize GLiNER2 model
let model = try await GLiNER2(
    manifestURL: URL(fileURLWithPath: "path/to/CoreMLArtifacts")
)

// Extract entities using Schema API
let text = "John Smith works at Apple Inc. in Cupertino."
let schema = Schema().entities(["person", "organization", "location"])
let result = try await model.extract(text: text, schema: schema, threshold: 0.5)

// Print results
for (label, entities) in result.entities {
    for entity in entities {
        print("\(label): \(entity.text) (score: \(entity.score))")
    }
}
```

### Using Convenience Methods

```swift
// Simple entity extraction
let entities = try await model.extractEntities(
    text: "Marie Curie discovered radium in Paris.",
    labels: ["person", "chemical", "location"],
    threshold: 0.5
)
```

## Usage

### Text Classification

```swift
// Multi-label classification with sigmoid
let schema = Schema()
    .classification(labels: ["technology", "business", "science"], multilabel: true)

let result = try await model.extract(text: "Apple announces new AI research.", schema: schema)
for (label, score) in result.classifications {
    print("\(label): \(score)")
}
```

### Structured Data Extraction

```swift
// Extract structured fields
let schema = Schema()
    .structure()
    .field("company", type: "organization")
    .field("location", type: "location")
    .field("person", type: "person")

let result = try await model.extract(text: text, schema: schema)
if let company = result.structuredData["company"]?.first {
    print("Company: \(company.text)")
}
```

### Custom Configuration

```swift
var config = Configuration.default
config.threshold = 0.7
config.maxSpanLength = 15
config.poolingMethod = .mean
config.similarityMetric = .cosine

let model = try await GLiNER2(
    manifestURL: manifestURL,
    config: config
)
```

### Per-Label Thresholds

```swift
var config = Configuration.default
config.labelThresholds = [
    "person": 0.6,
    "organization": 0.7,
    "location": 0.5
]

let model = try await GLiNER2(
    manifestURL: manifestURL,
    config: config
)
```

### Batch Processing

```swift
let texts = [
    "John works at Apple.",
    "Mary lives in Paris."
]

let results = try await model.batchExtract(
    texts: texts,
    schema: Schema().entities(["person", "organization", "location"]),
    threshold: 0.5
)
```

### Using Custom Models

```swift
// Load from custom manifest
let manifestURL = URL(fileURLWithPath: "/path/to/export_manifest.json")
let model = try await GLiNER2(manifestURL: manifestURL)

// Or specify individual components
let model = try await GLiNER2(
    encoderURL: URL(fileURLWithPath: "/path/to/encoder.mlpackage"),
    spanScorerURL: URL(fileURLWithPath: "/path/to/scorer.mlpackage"),
    tokenizerURL: URL(fileURLWithPath: "/path/to/tokenizer")
)
```

## Architecture

GLiNERSwift implements the complete GLiNER2 pipeline:

1. **Core ML Models**
    - **Encoder**: DeBERTa-v3 backbone converted to Core ML
    - **Span Scorer**: BiLSTM + projection layers for entity scoring
    - **Count Predictor**: Entity count estimation (GLiNER2 feature)
    - **Schema Projector**: Multi-task head for classification/structured extraction
    - **Classifier**: Sigmoid/softmax classification head

2. **Custom Tokenizer**
    - SentencePiece Unigram tokenizer (pure Swift, no dependencies)
    - GLiNER2-specific prompt formatting with special tokens
    - Schema encoding for multi-task extraction

3. **Swift Runtime**
    - Schema builder API matching Python GLiNER2
    - Span decoding with NMS and greedy selection
    - Batch processing and async/await support

## Converting Custom Models

To convert your own fine-tuned GLiNER2 model:

```bash
cd Scripts
pip install -r requirements.txt
python convert_to_coreml.py \
    --model urchade/gliner_multi-v2.1 \
    --output ../CoreMLArtifacts \
    --precision float16
```

This will:
1. Download the GLiNER2 model from HuggingFace
2. Convert all components to Core ML:
   - Encoder (DeBERTa backbone)
   - Span representation model
   - Count predictor
   - Schema projector
   - Classifier head
3. Export tokenizer files
4. Create `export_manifest.json` with all model paths
5. Save everything to the output directory

Options:
- `--model`: HuggingFace model ID or local path
- `--output`: Output directory for Core ML models
- `--precision`: `float16` (default, smaller) or `float32` (more precise)
- `--max-seq-len`: Maximum sequence length (default: 384)
- `--device`: `cpu` or `cuda` for conversion

## Testing

### Running Tests

```bash
swift test
```

### Generating Test Fixtures

The test suite validates parity with the Python GLiNER2 implementation:

```bash
cd Scripts
python generate_test_fixtures.py
```

This creates `Tests/GLiNERSwiftTests/Fixtures/python_outputs.json` with:
- Tokenizer outputs
- Encoder hidden states
- Expected entity predictions

Tests compare Swift outputs against these fixtures with appropriate tolerances.

## Benchmarks

Compare Python GLiNER2 vs Swift GLiNERSwift performance:

```bash
# Run both Python and Swift benchmarks
cd Scripts
python compare_benchmarks.py \
    --fixtures test_fixtures.jsonl \
    --iterations 10 \
    --warmup 2 \
    --python-model urchade/gliner_multi-v2.1

# Or run individually
# Swift only
swift run -c release gliner-benchmarks \
    --fixtures Scripts/test_fixtures.jsonl \
    --iterations 10 \
    --warmup 2

# Python only
cd Scripts
python benchmarks.py \
    --fixtures test_fixtures.jsonl \
    --model urchade/gliner_multi-v2.1 \
    --iterations 10 \
    --warmup 2
```

Both runners output latency statistics (average/median/p95) and throughput (characters/sec, requests/sec).

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `poolingMethod` | `.mean` | How to pool token embeddings into spans |
| `similarityMetric` | `.cosine` | Metric for span-label similarity |
| `threshold` | `0.5` | Global confidence threshold |
| `maxSequenceLength` | `512` | Maximum tokens per input |
| `strideLength` | `256` | Overlap for sliding window |
| `maxSpanLength` | `10` | Maximum span length in tokens |
| `useWordBoundaries` | `true` | Respect word boundaries |
| `nmsThreshold` | `0.5` | IoU threshold for NMS |
| `labelThresholds` | `[:]` | Per-label threshold overrides |

## Performance

Current benchmarks (3 samples, 2 iterations, CPU):
- **Swift GLiNERSwift**: ~2170ms average latency, 26.7 chars/sec
- **Python GLiNER v1**: ~60ms average latency, 974 chars/sec

Swift is currently ~36x slower due to:
- CPU-only implementation (Python uses optimized PyTorch/ONNX)
- Focus on correctness and feature parity over optimization
- No ANE support yet (dynamic prompt shapes)

Future optimization opportunities:
- ANE support when Core ML accepts dynamic shapes
- Metal compute kernels for scoring
- Batch processing optimizations
- Model quantization

## Requirements

- iOS 16.0+ / macOS 13.0+
- Swift 5.9+
- Xcode 15.0+

## Dependencies

- [swift-numerics](https://github.com/apple/swift-numerics) - Numerical operations

Note: GLiNERSwift includes a custom pure-Swift SentencePiece tokenizer implementation. No external tokenization dependencies required.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This project is based on and incorporates concepts from:
- [GLiNER2](https://github.com/fastino-ai/GLiNER2) (Apache 2.0)
- [GLiNER](https://github.com/urchade/GLiNER) (Apache 2.0)

See [NOTICE](NOTICE) for full attribution.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- [GLiNER](https://github.com/urchade/GLiNER) - Original GLiNER implementation by Urchade Zaratiana
- [GLiNER2](https://github.com/fastino-ai/GLiNER2) - GLiNER2 implementation by Fastino AI
- The GLiNER2 model architecture and training methodology

## Citation

If you use GLiNERSwift in your research, please cite the original GLiNER paper:

```bibtex
@article{gliner2023,
  title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
  author={Urchade Zaratiana and others},
  journal={arXiv preprint arXiv:2311.08526},
  year={2023}
}
```

## Support

- 🐛 [Issue Tracker](https://github.com/tomaarsen/GLiNERSwift/issues)
- 💬 [Discussions](https://github.com/tomaarsen/GLiNERSwift/discussions)
- 📚 See `FEATURE_PARITY_SUMMARY.md` for implementation details
