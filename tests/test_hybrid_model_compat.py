import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_model import build_model
from models.transformer_encoder import ContextualPositionalEncoding


def test_mask_context_channels_tolerates_legacy_checkpoint_attrs():
    model = build_model(
        d_model=64,
        n_heads=4,
        n_layers=2,
        input_dim=32,
        noise_dim=16,
        output_dim=3,
        dropout_p=0.1,
    )
    model.eval()
    src = torch.randn(1, 168, 32)

    delattr(model, "pe_mode")
    delattr(model, "context_dim")
    delattr(model.transformer_encoder, "pe_mode")
    delattr(model.transformer_encoder.pos_encoder, "mode")
    delattr(model, "film")

    masked = model._mask_context_channels(src)
    assert masked.shape == src.shape
    out = model(src)
    assert out.shape[0] == src.shape[0]


def test_contextual_pe_tolerates_missing_inner_module():
    pe = ContextualPositionalEncoding(d_model=8, context_dim=2, dropout=0.1)
    x = torch.randn(1, 4, 8)
    delattr(pe, "_inner")
    out = pe(x, context=torch.randn(1, 4, 2))
    assert out.shape == x.shape
