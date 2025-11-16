#!/usr/bin/env python3
"""
Alternative approach: Use a standard BERT encoder that converts cleanly to Core ML.
We'll use microsoft/deberta-v3-small which is similar architecture to GLiNER's encoder.
"""

import os
import torch
import numpy as np
import coremltools as ct
from transformers import AutoModel

print("=" * 60)
print("Alternative Encoder to Core ML")
print("=" * 60)

# Configuration
MODEL_NAME = "bert-base-uncased"  # Standard BERT, should convert cleanly
OUTPUT_DIR = "../Sources/GLiNERSwift/Resources"
COREML_PATH = os.path.join(OUTPUT_DIR, "GLiNEREncoder.mlpackage")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n1. Loading BERT model: {MODEL_NAME}")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print(f"   ✓ Model loaded successfully")
print(f"   ✓ Hidden size: {model.config.hidden_size}")

# Wrapper for clean interface
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

print("\n2. Creating encoder wrapper...")
wrapped_encoder = EncoderWrapper(model)
wrapped_encoder.eval()

# Test inputs
batch_size = 1
seq_len = 16
dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len), dtype=torch.long)
dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

print("\n3. Testing wrapped encoder...")
with torch.no_grad():
    test_output = wrapped_encoder(dummy_input_ids, dummy_attention_mask)
    print(f"   ✓ Output shape: {test_output.shape}")
    print(f"   ✓ Output dtype: {test_output.dtype}")

print("\n4. Tracing model with TorchScript...")
with torch.no_grad():
    traced_model = torch.jit.trace(
        wrapped_encoder,
        (dummy_input_ids, dummy_attention_mask)
    )
    traced_output = traced_model(dummy_input_ids, dummy_attention_mask)
    print(f"   ✓ Traced output shape: {traced_output.shape}")

print("\n5. Converting to Core ML...")
try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="hidden_states")
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to='mlprogram'
    )
    
    mlmodel.save(COREML_PATH)
    print(f"   ✓ Core ML model saved to: {COREML_PATH}")
    
    print("\n" + "=" * 60)
    print("✓ Conversion completed successfully!")
    print("=" * 60)
    print(f"\nCore ML model: {COREML_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Hidden dimension: {model.config.hidden_size}")
    print(f"\nModel inputs:")
    print(f"  - input_ids: [1, 1-512] (Int32)")
    print(f"  - attention_mask: [1, 1-512] (Int32)")
    print(f"Model output:")
    print(f"  - hidden_states: [1, sequence, {model.config.hidden_size}] (Float32)")
    print(f"\nNote: This encoder produces {model.config.hidden_size}-dim embeddings.")
    print(f"      GLiNER uses 512-dim. You may need to add a projection layer")
    print(f"      or adjust the Swift code to handle {model.config.hidden_size} dims.")
    
except Exception as e:
    print(f"\n✗ Core ML conversion failed: {e}")
    import traceback
    traceback.print_exc()
    raise
