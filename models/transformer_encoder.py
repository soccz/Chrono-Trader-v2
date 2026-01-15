
import torch
import torch.nn as nn
import math
from utils.config import config

class ContextualPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, context_dim=0, dropout=0.1):
        super(ContextualPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Static PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

        # Dynamic Context Projection
        self.context_dim = context_dim
        if context_dim > 0:
            # Maps context (e.g., market index return) to d_model space to modulate time perception
            self.context_proj = nn.Linear(context_dim, d_model)

    def forward(self, x, context=None):
        # x: (batch, seq_len, d_model)
        # context: (batch, seq_len, context_dim)
        
        # Add static PE
        x = x + self.pe[:, :x.size(1), :]
        
        # Add dynamic context PE if available
        if self.context_dim > 0 and context is not None:
             # Project context to d_model size
             # context_emb: (batch, seq_len, d_model)
             context_emb = self.context_proj(context)
             
             # We add the context embedding to the sequence embedding
             # This effectively shifts the representation of "time" based on market state
             x = x + context_emb
             
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
    
    def forward(self, src, return_attention=False):
        # Self-attention with optional attention weight return
        src2, attn_weights = self.self_attn(src, src, src, need_weights=return_attention)
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
    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1, context_dim=0, use_causal_mask=False):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.use_causal_mask = use_causal_mask
        self.encoder_layer = nn.Linear(input_dim, d_model)
        
        # Use the new Contextual PE
        self.pos_encoder = ContextualPositionalEncoding(d_model, context_dim=context_dim, dropout=dropout)
        
        # Use custom layers that can return attention
        self.layers = nn.ModuleList([
            AttentionExtractorEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_layer.weight.data.uniform_(-initrange, initrange)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate a causal mask to prevent attending to future positions."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, context=None, return_attention=False):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.encoder_layer(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src, context=context)
        
        attention_weights = []
        for layer in self.layers:
            if return_attention:
                src, attn = layer(src, return_attention=True)
                attention_weights.append(attn)
            else:
                src = layer(src, return_attention=False)
        
        if return_attention:
            return src, attention_weights
        return src

def build_transformer_encoder(input_dim, d_model, n_heads, n_layers, dropout_p, context_dim=0, use_causal_mask=False):
    return TransformerEncoder(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout_p,
        context_dim=context_dim,
        use_causal_mask=use_causal_mask
    )

