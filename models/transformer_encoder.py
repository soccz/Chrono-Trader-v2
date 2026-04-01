
import torch
import torch.nn as nn
import math
from utils.config import config


class ContextualPositionalEncoding(nn.Module):
    """Legacy alias — kept for backward compat. Use CyclePE directly."""
    def __init__(self, d_model, max_len=5000, context_dim=0, dropout=0.1):
        super().__init__()
        self._inner = CyclePE(d_model, max_len=max_len, context_dim=context_dim,
                              dropout=dropout, mode='cycle_pe')

    def forward(self, x, context=None):
        inner = getattr(self, "_inner", None)
        if inner is not None:
            return inner(x, context=context)

        # Legacy checkpoint fallback: some older serialized objects may carry the
        # old wrapper type without the nested `_inner` module.
        pe = getattr(self, "pe", None)
        if isinstance(pe, torch.Tensor):
            x = x + pe[:, :x.size(1), :]

        dropout = getattr(self, "dropout", None)
        if isinstance(dropout, nn.Module):
            return dropout(x)
        return x


class CyclePE(nn.Module):
    """
    Cycle-aware Positional Encoding (paper §4.3).

    mode:
      'static'    — standard sinusoidal PE only (ablation baseline)
      'concat_a'  — state features appended as extra input channels (ablation A)
                    NOTE: in concat_a mode the PE itself is static; state injection
                    happens at the input projection level in TransformerEncoder.
      'cycle_pe'  — sinusoidal PE + intensity projected into d_model space (H2 main)
      'cycle_pe_full' — sinusoidal PE + intensity + position both projected (appendix)

    context layout (context_dim must match):
      context[:, :, 0] = market_index_return  (position proxy)
      context[:, :, 1] = historical_similarity (intensity proxy)
    """
    INTENSITY_IDX = 1   # cycle_intensity channel in context
    POSITION_IDX  = 0   # cycle_position channel in context

    def __init__(self, d_model, max_len=5000, context_dim=0, dropout=0.1,
                 mode='cycle_pe'):
        super().__init__()
        assert mode in ('static', 'concat_a', 'cycle_pe', 'cycle_pe_full'), \
            f"Unknown CyclePE mode: {mode}"
        self.mode = mode
        self.context_dim = context_dim
        self.dropout = nn.Dropout(p=dropout)

        # Static sinusoidal PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        # State projections (only for cycle_pe / cycle_pe_full)
        if mode in ('cycle_pe', 'cycle_pe_full') and context_dim > 0:
            # intensity → d_model  (always used in cycle_pe*)
            self.intensity_proj = nn.Linear(1, d_model)
        if mode == 'cycle_pe_full' and context_dim > 0:
            # position → d_model  (appendix variant)
            self.position_proj = nn.Linear(1, d_model)

    def forward(self, x, context=None):
        """
        x       : (B, T, d_model)
        context : (B, T, context_dim)  — required for cycle_pe* modes
        returns : (B, T, d_model)
        """
        mode = getattr(self, "mode", "cycle_pe_full")
        x = x + self.pe[:, :x.size(1), :]

        if mode in ('cycle_pe', 'cycle_pe_full') and context is not None:
            intensity = context[:, :, self.INTENSITY_IDX:self.INTENSITY_IDX+1]  # (B,T,1)
            x = x + self.intensity_proj(intensity)

        if mode == 'cycle_pe_full' and context is not None:
            position = context[:, :, self.POSITION_IDX:self.POSITION_IDX+1]    # (B,T,1)
            x = x + self.position_proj(position)

        return self.dropout(x)


class AttentionExtractorEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer that can return attention weights.
    Wraps nn.MultiheadAttention for attention extraction.
    """
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(AttentionExtractorEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src, return_attention=False, attn_mask=None):
        # Self-attention with optional attention weight return
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=attn_mask, need_weights=return_attention
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        if return_attention:
            return src, attn_weights
        return src


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with pluggable CyclePE.

    pe_mode:
      'static'        — sinusoidal only
      'concat_a'      — state features concatenated to input before projection
      'cycle_pe'      — intensity injected via PE (H2 main)
      'cycle_pe_full' — intensity + position injected via PE (appendix)
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1,
                 context_dim=0, use_causal_mask=False, pe_mode='cycle_pe'):
        super().__init__()
        self.d_model = d_model
        self.use_causal_mask = use_causal_mask
        self.pe_mode = pe_mode

        # Concat-A: state features are appended to raw input before projection
        proj_input_dim = input_dim + context_dim if pe_mode == 'concat_a' else input_dim
        self.encoder_layer = nn.Linear(proj_input_dim, d_model)

        self.pos_encoder = CyclePE(d_model, context_dim=context_dim,
                                   dropout=dropout, mode=pe_mode)

        self.layers = nn.ModuleList([
            AttentionExtractorEncoderLayer(d_model, n_heads,
                                           dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(n_layers)
        ])

        initrange = 0.1
        self.encoder_layer.weight.data.uniform_(-initrange, initrange)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, src, context=None, return_attention=False):
        # src: (B, T, input_dim),  context: (B, T, context_dim)
        pe_mode = getattr(self, "pe_mode", getattr(config.Gan, "PE_MODE", "cycle_pe_full"))
        if pe_mode == 'concat_a' and context is not None:
            src = torch.cat([src, context], dim=-1)  # (B, T, input_dim+context_dim)

        src = self.encoder_layer(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src, context=context)

        use_causal_mask = bool(getattr(self, "use_causal_mask", False))
        attn_mask = self._causal_mask(src.size(1), src.device) if use_causal_mask else None

        attention_weights = []
        for layer in self.layers:
            if return_attention:
                src, attn = layer(src, return_attention=True, attn_mask=attn_mask)
                attention_weights.append(attn)
            else:
                src = layer(src, return_attention=False, attn_mask=attn_mask)

        if return_attention:
            return src, attention_weights
        return src


def build_transformer_encoder(input_dim, d_model, n_heads, n_layers, dropout_p,
                              context_dim=0, use_causal_mask=True, pe_mode='cycle_pe'):
    return TransformerEncoder(
        input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dropout=dropout_p, context_dim=context_dim,
        use_causal_mask=use_causal_mask, pe_mode=pe_mode,
    )
