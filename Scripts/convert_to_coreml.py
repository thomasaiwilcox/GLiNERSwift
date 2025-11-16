#!/usr/bin/env python3
"""GLiNER2 exporter that emits Core ML artifacts for the Swift runtime."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from gliner2.model import Extractor


LOGGER = logging.getLogger("gliner2_export")


def patch_deberta_scaled_sqrt() -> callable:
    """Work around DeBERTa's int sqrt tracing bug if that encoder is used."""

    try:
        from transformers.models import deberta_v2
    except Exception:  # transformers optional
        return lambda: None

    modeling_deberta_v2 = deberta_v2.modeling_deberta_v2
    original_scaled = getattr(modeling_deberta_v2, "scaled_size_sqrt", None)
    original_build_rpos = getattr(modeling_deberta_v2, "build_rpos", None)
    if original_scaled is None:
        return lambda: None

    def patched(query_layer: torch.Tensor, scale_factor: int):
        head_dim = query_layer.size(-1)
        scale_value = (head_dim * scale_factor) ** 0.5
        return torch.tensor(scale_value, dtype=query_layer.dtype, device=query_layer.device)

    modeling_deberta_v2.scaled_size_sqrt = patched

    def patched_build_rpos(query_layer, key_layer, relative_pos, position_buckets, max_relative_positions):
        return relative_pos

    if original_build_rpos is not None:
        modeling_deberta_v2.build_rpos = patched_build_rpos

    def restore():
        modeling_deberta_v2.scaled_size_sqrt = original_scaled
        if original_build_rpos is not None:
            modeling_deberta_v2.build_rpos = original_build_rpos

    return restore


class EncoderWrapper(torch.nn.Module):
    """Expose the encoder backbone as a Core ML friendly module."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long())
        return outputs.last_hidden_state


class SpanRepWrapper(torch.nn.Module):
    """Adapter around the SpanRepLayer."""

    def __init__(self, span_rep: torch.nn.Module):
        super().__init__()
        self.span_rep = span_rep

    def forward(self, token_embeddings: torch.Tensor, span_indices: torch.Tensor) -> torch.Tensor:
        return self.span_rep(token_embeddings, span_indices.long())


class ClassifierWrapper(torch.nn.Module):
    """Standalone classification head."""

    def __init__(self, classifier: torch.nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, schema_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(schema_embeddings)


class CountPredictorWrapper(torch.nn.Module):
    """Predicts hierarchical counts from prompt embeddings."""

    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        return self.predictor(prompt_embeddings)


class CountEmbedWrapper(torch.nn.Module):
    """Generates count-aware structure projections for span scoring."""

    def __init__(self, embedder: torch.nn.Module):
        super().__init__()
        self.embedder = embedder
        self.max_count = getattr(embedder, "max_count", 20)

    def forward(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        return self.embedder(label_embeddings, self.max_count)


def _clone_linear(linear: torch.nn.Linear) -> torch.nn.Linear:
    clone = torch.nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        clone.weight.copy_(linear.weight)
        if linear.bias is not None:
            clone.bias.copy_(linear.bias)
    return clone


def _clone_layer_norm(norm: torch.nn.LayerNorm) -> torch.nn.LayerNorm:
    clone = torch.nn.LayerNorm(norm.normalized_shape, eps=norm.eps)
    with torch.no_grad():
        clone.weight.copy_(norm.weight)
        clone.bias.copy_(norm.bias)
    return clone


class ExportableSelfAttention(torch.nn.Module):
    def __init__(self, attention: torch.nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = getattr(attention, "batch_first", False)
        in_proj = attention.in_proj_weight
        in_bias = attention.in_proj_bias
        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        with torch.no_grad():
            self.q_proj.weight.copy_(in_proj[:self.embed_dim])
            self.k_proj.weight.copy_(in_proj[self.embed_dim:2 * self.embed_dim])
            self.v_proj.weight.copy_(in_proj[2 * self.embed_dim:])
            self.q_proj.bias.copy_(in_bias[:self.embed_dim])
            self.k_proj.bias.copy_(in_bias[self.embed_dim:2 * self.embed_dim])
            self.v_proj.bias.copy_(in_bias[2 * self.embed_dim:])
        self.out_proj = _clone_linear(attention.out_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            batch, seq_len, _ = x.shape
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, v)
            context = context.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.embed_dim)
            return self.out_proj(context)

        # Fallback for legacy seq-first cases
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        seq_len, batch, _ = q.shape
        q = q.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.permute(2, 0, 1, 3).contiguous().view(seq_len, batch, self.embed_dim)
        return self.out_proj(context)


class ExportableTransformerLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.TransformerEncoderLayer):
        super().__init__()
        self.self_attn = ExportableSelfAttention(layer.self_attn)
        self.norm1 = _clone_layer_norm(layer.norm1)
        self.norm2 = _clone_layer_norm(layer.norm2)
        self.linear1 = _clone_linear(layer.linear1)
        self.linear2 = _clone_linear(layer.linear2)
        self.dropout = torch.nn.Dropout(layer.dropout.p)
        self.dropout1 = torch.nn.Dropout(layer.dropout1.p)
        self.dropout2 = torch.nn.Dropout(layer.dropout2.p)
        self.activation = layer.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class ExportableDownscaledTransformer(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_projector = _clone_linear(original.in_projector)
        self.layers = torch.nn.ModuleList([
            ExportableTransformerLayer(layer) for layer in original.transformer.layers
        ])
        self.out_projector = torch.nn.Sequential(*[
            module if not isinstance(module, torch.nn.Linear) else _clone_linear(module)
            for module in original.out_projector
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        x = self.in_projector(x)
        for layer in self.layers:
            x = layer(x)
        x = torch.cat([x, original_x], dim=-1)
        return self.out_projector(x)


def patch_count_embed_transformer(model: Extractor) -> None:
    embedder = getattr(model, "count_embed", None)
    if embedder is None:
        return
    transformer = getattr(embedder, "transformer", None)
    if transformer is None:
        return
    embedder.transformer = ExportableDownscaledTransformer(transformer)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GLiNER2 modules to Core ML")
    parser.add_argument("--model-id", default="fastino/gliner2-base-v1", help="Hugging Face model ID or local path")
    parser.add_argument("--output-dir", default="CoreMLArtifacts", help="Directory where export artifacts are written")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum token sequence length")
    parser.add_argument("--max-schema-tokens", type=int, default=64, help="Maximum tokens per schema prompt")
    parser.add_argument("--compute-units", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"], default="ALL",
                        help="Preferred Core ML compute target")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16", help="Weight precision for mlpackages")
    parser.add_argument("--skip-span-rep", action="store_true", help="Skip exporting the span representation head")
    parser.add_argument("--skip-classifier", action="store_true", help="Skip exporting the classification head")
    parser.add_argument("--skip-count-predictor", action="store_true", help="Skip exporting the count predictor head")
    parser.add_argument("--skip-count-embed", action="store_true", help="Skip exporting the count embedding head")
    parser.add_argument("--export-tokenizer", action="store_true", help="Also dump the tokenizer files next to the models")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_unit(label: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping[label]


def trace_module(module: torch.nn.Module, example_inputs: tuple[torch.Tensor, ...]) -> torch.jit.ScriptModule:
    module.eval()
    module.cpu()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs)
    return traced


def convert_module(
    *,
    name: str,
    scripted: torch.jit.ScriptModule,
    inputs: List[ct.TensorType],
    destination: Path,
    compute_units: ct.ComputeUnit,
    precision: str,
) -> Path:
    LOGGER.info("Converting %s", name)
    mlmodel = ct.convert(
        scripted,
        convert_to="mlprogram",
        inputs=inputs,
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save(str(destination))
    return destination


def export_encoder(model: Extractor, args: argparse.Namespace, out_dir: Path, cu: ct.ComputeUnit) -> Path:
    wrapper = EncoderWrapper(model.encoder)
    seq_len = args.max_seq_len
    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
    scripted = trace_module(wrapper, (dummy_ids, dummy_mask))
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, seq_len)), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, seq_len)), dtype=np.int32),
    ]
    filename = "GLiNER2Encoder.mlpackage" if args.precision == "fp32" else "GLiNER2Encoder_fp16.mlpackage"
    return convert_module(
        name="encoder",
        scripted=scripted,
        inputs=inputs,
        destination=out_dir / filename,
        compute_units=cu,
        precision=args.precision,
    )


def export_span_rep(model: Extractor, args: argparse.Namespace, out_dir: Path, cu: ct.ComputeUnit) -> Path:
    wrapper = SpanRepWrapper(model.span_rep)
    hidden = model.encoder.config.hidden_size
    seq_len = args.max_seq_len
    max_width = model.config.max_width
    span_cap = seq_len * max_width
    dummy_tokens = torch.zeros(1, seq_len, hidden)
    dummy_spans = torch.zeros(1, span_cap, 2, dtype=torch.long)
    scripted = trace_module(wrapper, (dummy_tokens, dummy_spans))
    inputs = [
        # SpanRep reshapes back to [B, seq_len, max_width, hidden], so keep shapes static.
        ct.TensorType(name="token_embeddings", shape=(1, seq_len, hidden), dtype=np.float32),
        ct.TensorType(name="span_indices", shape=(1, span_cap, 2), dtype=np.int32),
    ]
    return convert_module(
        name="span_rep",
        scripted=scripted,
        inputs=inputs,
        destination=out_dir / "GLiNER2SpanRep.mlpackage",
        compute_units=cu,
        precision=args.precision,
    )


def export_classifier(model: Extractor, args: argparse.Namespace, out_dir: Path, cu: ct.ComputeUnit) -> Path:
    wrapper = ClassifierWrapper(model.classifier)
    hidden = model.encoder.config.hidden_size
    schema_cap = args.max_schema_tokens
    dummy = torch.zeros(schema_cap, hidden)
    scripted = trace_module(wrapper, (dummy,))
    inputs = [
        ct.TensorType(name="schema_embeddings", shape=(ct.RangeDim(1, schema_cap), hidden), dtype=np.float32),
    ]
    return convert_module(
        name="classifier",
        scripted=scripted,
        inputs=inputs,
        destination=out_dir / "GLiNER2Classifier.mlpackage",
        compute_units=cu,
        precision=args.precision,
    )


def export_count_predictor(model: Extractor, args: argparse.Namespace, out_dir: Path, cu: ct.ComputeUnit) -> Path:
    wrapper = CountPredictorWrapper(model.count_pred)
    hidden = model.encoder.config.hidden_size
    schema_cap = args.max_schema_tokens
    dummy = torch.zeros(schema_cap, hidden)
    scripted = trace_module(wrapper, (dummy,))
    inputs = [
        ct.TensorType(name="prompt_embeddings", shape=(ct.RangeDim(1, schema_cap), hidden), dtype=np.float32),
    ]
    return convert_module(
        name="count_predictor",
        scripted=scripted,
        inputs=inputs,
        destination=out_dir / "GLiNER2CountPredictor.mlpackage",
        compute_units=cu,
        precision=args.precision,
    )


def export_count_embed(model: Extractor, args: argparse.Namespace, out_dir: Path, cu: ct.ComputeUnit) -> Path:
    wrapper = CountEmbedWrapper(model.count_embed)
    hidden = model.encoder.config.hidden_size
    schema_cap = args.max_schema_tokens
    dummy = torch.zeros(schema_cap, hidden)
    scripted = trace_module(wrapper, (dummy,))
    inputs = [
        ct.TensorType(name="label_embeddings", shape=(ct.RangeDim(1, schema_cap), hidden), dtype=np.float32),
    ]
    return convert_module(
        name="count_embed",
        scripted=scripted,
        inputs=inputs,
        destination=out_dir / "GLiNER2CountEmbed.mlpackage",
        compute_units=cu,
        precision=args.precision,
    )


def export_tokenizer(model: Extractor, out_dir: Path) -> Optional[Path]:
    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is None:
        LOGGER.warning("Tokenizer not found on model.processor; skipping export")
        return None
    tokenizer_dir = out_dir / "tokenizer"
    ensure_dir(tokenizer_dir)
    tokenizer.save_pretrained(str(tokenizer_dir))
    return tokenizer_dir


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    restore_deberta = patch_deberta_scaled_sqrt()

    try:
        LOGGER.info("Loading GLiNER2 model %s", args.model_id)
        model = Extractor.from_pretrained(args.model_id)
        model.eval()
        patch_count_embed_transformer(model)

        out_dir = Path(args.output_dir)
        ensure_dir(out_dir)
        cu = compute_unit(args.compute_units)

        def relativize(path: Optional[Path]) -> Optional[str]:
            if path is None:
                return None
            try:
                return str(path.relative_to(out_dir))
            except ValueError:
                return str(path)

        artifacts: Dict[str, str] = {}
        artifacts["encoder"] = relativize(export_encoder(model, args, out_dir, cu))

        if not args.skip_span_rep:
            artifacts["span_rep"] = relativize(export_span_rep(model, args, out_dir, cu))

        if not args.skip_classifier:
            artifacts["classifier"] = relativize(export_classifier(model, args, out_dir, cu))

        if not args.skip_count_predictor:
            artifacts["count_predictor"] = relativize(export_count_predictor(model, args, out_dir, cu))

        if not args.skip_count_embed:
            artifacts["count_embed"] = relativize(export_count_embed(model, args, out_dir, cu))

        tokenizer_path = None
        if args.export_tokenizer:
            tokenizer_path = export_tokenizer(model, out_dir)

        metadata = {
            "model_id": args.model_id,
            "max_seq_len": args.max_seq_len,
            "max_schema_tokens": args.max_schema_tokens,
            "max_width": model.config.max_width,
            "hidden_size": model.encoder.config.hidden_size,
            "counting_layer": model.config.counting_layer,
            "max_count": getattr(model.count_embed, "max_count", 20),
            "precision": args.precision,
            "artifacts": artifacts,
            "tokenizer_dir": relativize(tokenizer_path),
            "model_config": model.config.to_dict(),
        }

        manifest_path = out_dir / "export_manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2))
        LOGGER.info("Export complete → %s", manifest_path)
        return 0
    finally:
        restore_deberta()


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
