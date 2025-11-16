# GLiNERSwift – Status (November 2025)

## Snapshot
- ✅ Prompt-aware tokenizer and greedy decoder run in Swift while both the encoder (`GLiNER2Encoder.mlpackage`) and the span scoring head (`GLiNERSpanScorer.mlpackage`) execute in Core ML.
- ✅ Span-head metadata still ships in `SpanHead/metadata.json`; scoring weights now live inside the Core ML package exported via `Scripts/convert_to_coreml.py`.
- ✅ `swift test` passes after regenerating fixtures and Core ML assets (`Scripts/convert_to_coreml.py`).
- ⚠️ ANE remains disabled; the encoder uses `MLComputeUnits.cpuAndGPU` because Core ML still rejects the dynamic prompt-aware sequence shapes we rely on.
- ⚙️ New macOS-only benchmarking harnesses (Swift + Python) share JSONL fixtures and serve as the basis for the upcoming performance study.
- 🔜 Sliding-window chunking must be reintroduced on top of the prompt-aware pipeline to recover long-document coverage.

## GLiNER2 Migration Plan (Tokenizer & Schema Path)
1. **Manifest-Driven Bootstrapping** – load `CoreMLArtifacts/export_manifest.json` at runtime, resolve encoder/span/classifier/count mlpackages plus the tokenizer folder, and thread those URLs through `GLiNERModel` initializers.
2. **Tokenizer Upgrade** – teach `SentencePieceUnigramModel`/`GLiNERTokenizer` to load SentencePiece assets from an arbitrary directory (not just SwiftPM resources) and register GLiNER2 special tokens (`[P]`, `[C]`, `[R]`, `[L]`, `[E]`, `[SEP_STRUCT]`, `[SEP_TEXT]`, `[EXAMPLE]`, `[OUTPUT]`, `[DESCRIPTION]`).
3. **Schema Formatter** – mirror `gliner2.processor.transform_single_record` + `format_input_with_mapping` for the entity-only path so the Swift runtime can generate schema token lists, `[selection]` prefixes for in-JSON classifications, and the subword/source mapping metadata needed later by the span/classifier heads.
4. **Pipeline Wiring** – once the tokenizer emits GLiNER2-compatible encodings, hook the new Core ML heads (span rep + classifier + count predictor) into `GLiNERModel` so entity extraction can follow the same steps as the Python `SchemaTransformer`.

This plan supersedes the legacy GLiNER1 prompt flow; all new work should happen strictly under the GLiNER2 schema contract described above.

## Completed Since Last Check-in
1. **Core ML Span Head** – Wrapped the BiLSTM + projection layers inside a standalone Core ML model (`GLiNERSpanScorer.mlpackage`) and updated `SpanScorer` to stream prompts/word embeddings through it asynchronously.
2. **Exporter Refresh** – `convert_to_coreml.py` now emits both the encoder and scorer with `--max-words/--max-prompts` limits plus skip flags for incremental rebuilds.
3. **macOS Benchmarking Tooling (Option A)**
   - `gliner-benchmarks` executable target (SwiftPM) with bundled sample fixtures.
   - `Scripts/benchmarks.py` for Python-side latency baselines using the HuggingFace GLiNER model.
   - `Scripts/prepare_benchmark_data.py` to curate or extend JSONL benchmark datasets.
4. **Manifest Loader Hook** – Added `GLiNERModel.init(manifestURL:)` plus a shared initializer path so the Swift runtime can boot directly from `CoreMLArtifacts/export_manifest.json` without hardcoding bundle resources.
5. **Manifest Consumers** – Benchmark CLI, unit tests, and the sample demo now construct `GLiNERModel` via the export manifest to ensure GLiNER2 artifacts are always loaded from disk.
6. **GLiNER2 Core ML Heads** – Introduced Swift wrappers for `GLiNER2SpanRep`, `GLiNER2Classifier`, and `GLiNER2CountPredictor` plus regression tests to prove the mlpackages load and produce outputs.

## Active Workstreams

### 1. macOS Benchmarking (CPU+GPU)
| Deliverable | Location | Status |
|-------------|----------|--------|
| Swift CLI runner with latency stats | `Sources/Benchmarks/main.swift` | ✅ Implemented (`gliner-benchmarks` product) |
| Shared sample fixtures | `Sources/Benchmarks/Resources/Fixtures/benchmark_samples.jsonl` | ✅ Seeded (50 entries) |
| Python parity benchmark | `Scripts/benchmarks.py` | ✅ Implemented (depends on `gliner` pip package) |
| Fixture prep script | `Scripts/prepare_benchmark_data.py` | ✅ Generates/extends JSONL data |
| Metrics to capture | avg / median / p95 latency, min/max, chars per second | ✅ Reported by both runners |
| Outstanding | Expand fixture set, add CI hooks, archive runs per device | ⏳

**Usage targets**
- Swift: `swift build --product gliner-benchmarks` followed by `swift run gliner-benchmarks --iterations 10 --output ./benchmark.json`
- Python: `pip install -r Scripts/requirements.txt` then `python Scripts/benchmarks.py --iterations 10 --output ./python_benchmark.json`

### 2. Sliding-Window Chunking Reintegration
Objective: bring back chunked inference for >512-token documents while preserving prompt-aware span scoring.

Planned steps:
1. **Tokenizer Pass** – extend `GLiNERTokenizer.encodePrompted` to optionally emit overlapping segments (window + stride metadata) keyed by prompt label order.
2. **Encoder Loop** – chunked encodings will be run sequentially (ANE-incompatible anyway) and stitched together, carrying forward prompt positions.
3. **Span Stitching** – update `SpanDecoder` to merge chunk-level spans using byte offsets + confidence-based tie-breaking (reuse ideas from legacy `ChunkProcessor`).
4. **Parity + Tests** – add fixtures under `Tests/GLiNERSwiftTests/Fixtures/Chunking` and regression tests that feed 1k+ token samples to ensure dedupe stability.
5. **Config Surface** – expose `Configuration.chunkingEnabled`, `chunkLength`, and `chunkStride` flags defaulting to “off” until validated.

## Constraints & Risks
- **Compute Units** – Only CPU+GPU compute units are supported; ANE activation will require refactoring token padding or adopting fixed prompt lengths.
- **Fixture Drift** – Whenever Python exporters regenerate weights, rerun both benchmark harnesses and `swift test` to keep parity.
- **Chunking Regression Risk** – Prompt-aware masks introduce additional bookkeeping versus the legacy cosine-similarity flow; exhaustive tests across overlapping predictions are required.

## Immediate Next Actions
1. Layer the new span representation / classifier / count-predictor Core ML wrappers on top of the GLiNER2 schema formatter so inference no longer depends on the legacy GLiNER1 span head.
2. Replace `SpanScorer` with the GLiNER2 span pipeline inside `GLiNERModel`, then validate chunking + decoder parity.
3. After the new pipeline runs end-to-end, rerun Swift/Python benchmarks with the GLiNER2 heads enabled and document the updated latency split (tokenize vs encode vs span rep vs classifier/count vs decode).

## Reference Commands
```bash
# Swift CLI benchmark
swift build --product gliner-benchmarks
swift run gliner-benchmarks --iterations 10 --warmup 2 --output ./benchmark-macos.json

# Python benchmark
cd Scripts
pip install -r requirements.txt
python benchmarks.py --iterations 10 --warmup 2 --output ../benchmark-python.json
```

---

Maintainers: keep this file updated after each exporter run, benchmark capture, or chunking milestone so downstream teams can track readiness.
