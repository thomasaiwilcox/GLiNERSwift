# GLiNER2 Runtime Migration Blueprint

This document tracks the Swift-side tasks required to run exported GLiNER2 Core ML artifacts. Each stage is mapped to the corresponding Python reference implementation under `gliner2/` for quick validation.

## 1. Resource Loading
- [x] Parse `CoreMLArtifacts/export_manifest.json` via `GLiNERManifest`.
- [x] Surface a Swift API (`GLiNERModel.init(manifestURL:)`) that resolves encoder/span_rep/classifier/count_predictor/tokenizer bundles at runtime.

## 2. Tokenizer & Schema Formatting
- [x] Allow `GLiNERTokenizer` to boot from an arbitrary tokenizer directory (validated with `TokenizerParityTests`).
- [x] Introduce `GLiNER2PromptConfiguration` + `GLiNER2SchemaEncoding` to describe the `[P]/[C]/[E]/[R]/[L]/[SEP_STRUCT]/[SEP_TEXT]` contract.
- [ ] Implement `SchemaTransformer` parity for entity-only inference:
  - Build classification prefixes and `[selection]` wrappers (`processor.py::build_classification_prefix`).
  - Generate deduplicated span structures + labels (`processor.py::infer_from_json`).
  - Produce schema tokens/text tokens/mappings identical to `format_input_with_mapping` (already partially covered by `GLiNERTokenizer.encodeGLiNER2SchemaInput`).

## 3. Core ML Heads
- [x] Wrap `GLiNER2SpanRep.mlpackage` (token embeddings → `[seq, width, hidden]`).
- [x] Wrap `GLiNER2Classifier.mlpackage` (prompt embeddings → logits for span filtering + classification tasks).
- [x] Wrap `GLiNER2CountPredictor.mlpackage` (prompt embeddings → hierarchical counts for JSON structures/relations).

Implementation notes:
- SpanRep wrapper accepts encoder hidden states + span indices/masks coming from the schema encoder, returning contiguous span embeddings (mirrors `gliner2/span_rep_head.py`).
- Classifier wrapper consumes schema/prompt embeddings and exposes the raw logits multi-array for downstream decoding.
- CountPredictor wrapper runs on prompt embeddings only and emits per-branch counts leveraged by JSON/relation decoding.

## 4. Swift Inference Pipeline
- [ ] Replace `SpanScorer` with GLiNER2 span pipeline:
  1. Encode schema/text input via tokenizer + encoder.
  2. Build span indices/masks; run SpanRep to obtain span embeddings.
  3. Apply classifier outputs + count predictions to filter spans by schema task.
  4. Convert span logits → probabilities → `Entity` values (reuse existing greedy decoder but with GLiNER2 semantics).
- [ ] Ensure `ChunkProcessor` + `TextChunker` still work once encoder inputs become schema-conditioned.

## 5. Validation & Tooling
- [ ] Add Swift unit tests mirroring Python `gliner2.processor.transform_single_record` outputs for curated fixtures.
- [x] Add smoke tests to ensure the exported GLiNER2 span/classifier/count heads load via manifest paths (`GLiNER2CoreMLHeadTests`).
- [ ] Create an integration test that feeds a manifest-exported GLiNER2 bundle through the new Swift pipeline and compares against Python engine outputs for the same samples.
- [ ] Update benchmarking harnesses to record GLiNER2-specific latency splits (legacy GLiNER1 path is deprecated and slated for removal).
- Stitch the three heads behind a new coordinator (e.g., `GLiNER2SpanPipeline`) that replaces the legacy `SpanScorer` inside `GLiNERModel` once feature parity is proven.
- Ensure the existing greedy decoder understands GLiNER2 logits/counts; refactor decoder inputs if necessary to keep entity outputs stable.

## 6. Integration Sequence (next sprint)
1. Land the three Core ML wrappers with unit tests that load dummy tensors/mocks to confirm I/O shapes.
2. Extend `GLiNERTokenizer.encodeGLiNER2SchemaInput` to emit span indices, masks, and prompt offsets expected by the new heads.
3. Introduce `GLiNER2SpanPipeline` and swap it into `GLiNERModel` behind a feature flag.
4. Add parity tests comparing Swift span/classifier/count outputs against `gliner2.processor` fixtures, then delete the legacy span scorer and assets.

Keep this checklist in sync as we land each milestone. Once sections (1)–(4) are complete we can remove the legacy span head + tokenizer assets entirely.
