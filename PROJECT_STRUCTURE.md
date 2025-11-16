# Project Structure

```
GLiNERSwift/
├── Package.swift                           # Swift package manifest
├── LICENSE                                 # MIT License
├── README.md                               # Main documentation
├── SETUP.md                               # Setup guide
├── .gitignore                             # Git ignore rules
├── .gitattributes                         # Git LFS configuration
│
├── Scripts/                               # Python tooling
│   ├── requirements.txt                   # Python dependencies
│   ├── convert_gliner_patched_scale.py   # Patched GLiNER → Core ML conversion
│   └── generate_test_fixtures.py         # Test data generation
│
├── Sources/GLiNERSwift/                   # Main Swift package
│   ├── GLiNERModel.swift                 # Public API
│   │
│   ├── Models/                           # Data models
│   │   ├── Entity.swift                  # Entity struct
│   │   ├── Configuration.swift           # Configuration options
│   │   └── GLiNEREncoder.swift          # Core ML wrapper
│   │
│   ├── Tokenization/                     # Tokenization layer
│   │   ├── GLiNERTokenizer.swift        # Tokenizer wrapper
│   │   └── TokenizedInput.swift         # Tokenization data structures
│   │
│   ├── Inference/                        # Inference pipeline
│   │   ├── SpanBuilder.swift            # Span construction
│   │   ├── LabelEncoder.swift           # Label encoding & caching
│   │   ├── SimilarityScorer.swift       # Similarity computation
│   │   ├── EntityExtractor.swift        # Entity extraction & NMS
│   │   └── ChunkProcessor.swift         # Sliding window merging
│   │
│   └── Resources/                        # Bundled resources
│       ├── README.md                     # Resources documentation
│       ├── GLiNER2Encoder.mlpackage/    # Core ML model (generated)
│       └── tokenizer/                    # Tokenizer files (generated)
│
├── Tests/GLiNERSwiftTests/               # Test suite
│   ├── TestFixtures.swift               # Fixture loading
│   ├── TokenizerParityTests.swift       # Tokenizer tests
│   ├── EncoderParityTests.swift         # Encoder tests
│   ├── EntityExtractionTests.swift      # End-to-end tests
│   └── Fixtures/                        # Test data
│       ├── README.md
│       └── python_outputs.json          # Generated from Python
│
└── Examples/                            # Example applications
    └── GLiNERDemo/                      # SwiftUI demo app
        ├── README.md
        └── GLiNERDemo/
            └── ContentView.swift        # Demo UI

```

## Key Components

### Python Scripts (`Scripts/`)
- **convert_gliner_patched_scale.py**: Converts GLiNER encoder to Core ML with patched DeBERTa ops
- **generate_test_fixtures.py**: Generates test data from Python implementation
- **requirements.txt**: Python dependencies (torch, coremltools, gliner)

### Swift Package (`Sources/GLiNERSwift/`)

#### Public API
- **GLiNERModel**: Main entry point for entity extraction

#### Models
- **Entity**: Represents extracted entities
- **Configuration**: Customizable extraction parameters
- **GLiNEREncoder**: Core ML model wrapper with async inference

#### Tokenization
- **GLiNERTokenizer**: HuggingFace tokenizer wrapper
- **TokenizedInput**: Tokenization result structures

#### Inference Pipeline
- **SpanBuilder**: Enumerates candidate spans, applies pooling (mean/max/concat)
- **LabelEncoder**: Encodes labels to embeddings with caching
- **SimilarityScorer**: Computes cosine/dot-product similarity
- **EntityExtractor**: Applies thresholds, NMS, converts to entities
- **ChunkProcessor**: Merges entities from overlapping chunks

### Tests (`Tests/`)
- **TokenizerParityTests**: Validates tokenization matches Python
- **EncoderParityTests**: Validates encoder outputs within tolerance
- **EntityExtractionTests**: End-to-end entity extraction validation
- **TestFixtures**: Loads Python-generated test data

### Examples
- **GLiNERDemo**: SwiftUI app demonstrating entity extraction

## Workflow

1. **Setup**: Run `Scripts/convert_gliner_patched_scale.py` to generate Core ML model
2. **Test**: Run `Scripts/generate_test_fixtures.py` to create test data
3. **Build**: `swift build` to compile the package
4. **Test**: `swift test` to validate parity with Python
5. **Use**: Import GLiNERSwift in your app

## Dependencies

### Runtime (Swift)
- swift-transformers: Tokenization
- swift-numerics: Numerical operations
- Accelerate: Vector operations (system framework)
- CoreML: Model inference (system framework)

### Build-time (Python)
- torch: PyTorch for model loading
- coremltools: Model conversion
- gliner: GLiNER2 model
- transformers: Tokenizer
- numpy: Array operations
