# GLiNERSwift Resources Directory

This directory contains Core ML models and tokenizer files needed for GLiNER2 inference.

## ⚠️ Models Not Included in Git

The Core ML model files (`.mlpackage` directories) are **NOT included in the git repository** to keep it lightweight. You must generate them using the conversion script.

## Required Files

After running the conversion script, this directory should contain:

- `GLiNER2Encoder.mlpackage/` - DeBERTa-v3 encoder (CPU/GPU optimized)
- `GLiNER2EncoderANE.mlpackage/` - DeBERTa-v3 encoder (ANE optimized, experimental)
- `GLiNERSpanScorer.mlpackage/` - Span scoring head (BiLSTM + projections)
- `SpanHead/` - Raw span head weights and metadata (*.bin files)
- `tokenizer/` - SentencePiece tokenizer configuration and vocabulary

**Total size:** ~1.1GB

## Generating Models

Run the conversion script from the repository root:

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
2. Convert all components to Core ML
3. Export tokenizer files
4. Save to `CoreMLArtifacts/` directory

Then copy or symlink the models to this Resources directory, OR use the manifest-based initialization that references `CoreMLArtifacts/` directly.

## Alternative: Use CoreMLArtifacts Directory

The Swift package can load models from `CoreMLArtifacts/` using the manifest:

```swift
let manifestURL = URL(fileURLWithPath: "CoreMLArtifacts/export_manifest.json")
let model = try await GLiNER2(manifestURL: manifestURL)
```

This is the recommended approach - no need to copy files to Resources.
