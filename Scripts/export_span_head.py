#!/usr/bin/env python3
"""Export GLiNER span head and prompt projection weights for the Swift runtime."""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def save_tensor(tensor: torch.Tensor, path: Path) -> str:
    array = tensor.detach().cpu().to(torch.float32).numpy()
    array.astype("<f4").tofile(path)
    return path.name


def save_linear(linear: torch.nn.Linear, prefix: str, out_dir: Path) -> Dict[str, object]:
    """Save a torch.nn.Linear's weights/bias to binary files."""
    weight_path = out_dir / f"{prefix}.weight.bin"
    bias_path = out_dir / f"{prefix}.bias.bin"

    weight_name = save_tensor(linear.weight, weight_path)
    bias_name = save_tensor(linear.bias, bias_path)

    return {
        "weight": weight_name,
        "bias": bias_name,
        "in_features": linear.in_features,
        "out_features": linear.out_features,
    }


def export_projection_layer(seq: torch.nn.Sequential, name: str, out_dir: Path) -> Dict[str, object]:
    """Export the two linear sublayers from a projection block."""
    # Layout: Linear -> ReLU -> Dropout -> Linear
    linear1 = seq[0]
    linear2 = seq[3]
    return {
        "fc1": save_linear(linear1, f"{name}_fc1", out_dir),
        "fc2": save_linear(linear2, f"{name}_fc2", out_dir),
    }


def export_lstm_direction(lstm: torch.nn.LSTM, layer_idx: int, reverse: bool, prefix: str, out_dir: Path) -> Dict[str, object]:
    suffix = f"l{layer_idx}_reverse" if reverse else f"l{layer_idx}"
    weight_ih = getattr(lstm, f"weight_ih_{suffix}")
    weight_hh = getattr(lstm, f"weight_hh_{suffix}")
    bias_ih = getattr(lstm, f"bias_ih_{suffix}")
    bias_hh = getattr(lstm, f"bias_hh_{suffix}")

    direction = "backward" if reverse else "forward"
    weight_ih_path = out_dir / f"{prefix}_{direction}.weight_ih.bin"
    weight_hh_path = out_dir / f"{prefix}_{direction}.weight_hh.bin"
    bias_path = out_dir / f"{prefix}_{direction}.bias.bin"

    weight_ih_name = save_tensor(weight_ih, weight_ih_path)
    weight_hh_name = save_tensor(weight_hh, weight_hh_path)
    bias_name = save_tensor(bias_ih + bias_hh, bias_path)

    return {
        "weight_ih": weight_ih_name,
        "weight_hh": weight_hh_name,
        "bias": bias_name,
    }


def export_lstm(lstm: torch.nn.LSTM, prefix: str, out_dir: Path) -> Dict[str, object]:
    if lstm.num_layers != 1:
        raise RuntimeError("Only single-layer LSTMs are supported")
    info = {
        "input_size": int(lstm.input_size),
        "hidden_size": int(lstm.hidden_size),
        "num_layers": int(lstm.num_layers),
        "bidirectional": bool(lstm.bidirectional),
        "forward": export_lstm_direction(lstm, 0, False, prefix, out_dir),
    }
    if lstm.bidirectional:
        info["backward"] = export_lstm_direction(lstm, 0, True, prefix, out_dir)
    return info


def main(model_name: str, output_dir: Path) -> None:
    from gliner import GLiNER  # Import lazily so script fails fast if missing

    print(f"Loading GLiNER model '{model_name}'...")
    model = GLiNER.from_pretrained(model_name)
    model.eval()

    span_layer = model.model.span_rep_layer.span_rep_layer
    prompt_layer = model.model.prompt_rep_layer
    rnn_layer = getattr(model.model, "rnn", None)
    if rnn_layer is None:
        raise RuntimeError("Model does not expose an RNN layer; expected LstmSeq2SeqEncoder")
    lstm = rnn_layer.lstm


    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting projection weights...")
    tokenizer = model.data_processor.transformer_tokenizer

    metadata = {
        "model": model_name,
        "hidden_size": int(model.config.hidden_size),
        "max_width": int(model.config.max_width),
        "class_token_index": int(model.config.class_token_index),
        "ent_token": model.config.ent_token,
        "sep_token": model.config.sep_token,
        "special_tokens": {
            "flert_token": "[FLERT]",
            "flert_token_index": int(tokenizer.convert_tokens_to_ids("[FLERT]")),
            "ent_token_index": int(tokenizer.convert_tokens_to_ids(model.config.ent_token)),
            "sep_token_index": int(tokenizer.convert_tokens_to_ids(model.config.sep_token)),
            "mask_token_index": int(tokenizer.mask_token_id),
            "cls_token_index": int(tokenizer.cls_token_id),
            "base_sep_token_index": int(tokenizer.sep_token_id),
            "pad_token_index": int(tokenizer.pad_token_id),
        },
        "layers": {
            "project_start": export_projection_layer(span_layer.project_start, "project_start", output_dir),
            "project_end": export_projection_layer(span_layer.project_end, "project_end", output_dir),
            "out_project": export_projection_layer(span_layer.out_project, "out_project", output_dir),
            "prompt_projection": export_projection_layer(prompt_layer, "prompt_projection", output_dir),
        },
        "rnn": export_lstm(lstm, "rnn", output_dir),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Export complete. Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GLiNER span head weights for Swift runtime")
    parser.add_argument("--model", default="urchade/gliner_small-v2.1", help="HuggingFace model name or local path")
    parser.add_argument(
        "--output",
        default="../Sources/GLiNERSwift/Resources/SpanHead",
        help="Directory where binary weight files and metadata.json will be written",
    )
    args = parser.parse_args()

    main(args.model, Path(args.output).resolve())
