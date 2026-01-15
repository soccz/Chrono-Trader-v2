
import torch
import torch.nn as nn
import numpy as np
from utils.config import config
from utils.image_converter import to_gaf_image, batch_to_gaf_tensor
from models.transformer_encoder import build_transformer_encoder
from models.gan_decoder import build_gan_decoder

class ExplainableGatedFusion(nn.Module):
    """
    Explainable Gated Fusion: Fuses Global (Transformer) and Local (CNN) features
    using explicit pattern similarity scores.
    
    Instead of a pure black-box gate, this module:
    1. Projects both Transformer and CNN outputs to a common space
    2. Compares each to a learned "prototype bank" (patterns learned during training)
    3. Uses the similarity scores to inform the fusion gate
    
    This makes the gating decision explainable:
    - High transformer_similarity → Transformer's pattern matches historical successes
    - High cnn_similarity → CNN's pattern matches historical successes
    - Gate favors the path with higher similarity
    """
    def __init__(self, d_model, cnn_dim, n_prototypes=16):
        super(ExplainableGatedFusion, self).__init__()
        
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        
        # Project CNN output to d_model for fair comparison
        self.cnn_proj = nn.Linear(cnn_dim, d_model) if cnn_dim != d_model else nn.Identity()
        
        # Learned prototype bank: representative patterns from training
        # These are learned end-to-end and represent "what good patterns look like"
        self.prototype_bank = nn.Parameter(torch.randn(n_prototypes, d_model))
        nn.init.xavier_uniform_(self.prototype_bank)
        
        # Similarity aggregation: converts (n_prototypes,) similarities to a scalar
        self.transformer_sim_agg = nn.Sequential(
            nn.Linear(n_prototypes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.cnn_sim_agg = nn.Sequential(
            nn.Linear(n_prototypes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Final gate: combines similarity info with raw features
        # Input: [transformer_features, cnn_features, transformer_sim, cnn_sim]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model + d_model + 2, d_model),
            nn.Sigmoid()
        )

    def _compute_similarity(self, features, aggregator):
        """
        Compute cosine similarity between features and prototype bank.
        
        Args:
            features: (batch, d_model)
            aggregator: similarity aggregation network
        Returns:
            similarity_score: (batch, 1)
        """
        # Normalize for cosine similarity
        features_norm = nn.functional.normalize(features, p=2, dim=1)  # (batch, d_model)
        prototypes_norm = nn.functional.normalize(self.prototype_bank, p=2, dim=1)  # (n_prototypes, d_model)
        
        # Cosine similarity with all prototypes: (batch, n_prototypes)
        similarities = torch.mm(features_norm, prototypes_norm.t())
        
        # Aggregate to single score
        similarity_score = aggregator(similarities)  # (batch, 1)
        
        return similarity_score, similarities

    def forward(self, global_features, local_features, return_gate_info=False):
        """
        Args:
            global_features: (batch, d_model) - Transformer encoder output
            local_features: (batch, cnn_dim) - CNN encoder output
            return_gate_info: if True, return explainability info
        Returns:
            fused: (batch, d_model)
            gate_info: dict with explainability info (transformer_sim, cnn_sim, gate_values) - only if return_gate_info=True
        """
        # Project CNN to d_model
        local_proj = self.cnn_proj(local_features)  # (batch, d_model)
        
        # Compute explicit pattern similarities
        transformer_sim, transformer_proto_sims = self._compute_similarity(
            global_features, self.transformer_sim_agg
        )  # (batch, 1)
        cnn_sim, cnn_proto_sims = self._compute_similarity(
            local_proj, self.cnn_sim_agg
        )  # (batch, 1)
        
        # Combine features with similarity scores for gating decision
        combined = torch.cat([
            global_features, 
            local_proj, 
            transformer_sim, 
            cnn_sim
        ], dim=1)  # (batch, d_model + d_model + 2)
        
        gate = self.gate_net(combined)  # (batch, d_model)
        
        # Weighted fusion
        fused = gate * global_features + (1 - gate) * local_proj
        
        if return_gate_info:
            gate_info = {
                'transformer_similarity': transformer_sim.detach().cpu().numpy(),
                'cnn_similarity': cnn_sim.detach().cpu().numpy(),
                'transformer_proto_sims': transformer_proto_sims.detach().cpu().numpy(),
                'cnn_proto_sims': cnn_proto_sims.detach().cpu().numpy(),
                'gate_values': gate.mean(dim=1).detach().cpu().numpy(),  # Average gate for interpretability
                'prototype_bank': self.prototype_bank.detach().cpu().numpy()
            }
            return fused, gate_info
        
        return fused


# Keep old class name as alias for backward compatibility
GatedFusion = ExplainableGatedFusion

class HybridModel(nn.Module):
    """
    Enhanced Hybrid Model with Contextual Positional Encoding and Gated Fusion.
    
    1. Contextual PE: Injects Market Index Return (BTC/ETH) into the Transformer's time perception.
    2. Gated Fusion: Dynamically balances between Transformer (Global) and CNN (Local) features.
    """
    def __init__(self, d_model, n_heads, n_layers, input_dim, noise_dim, output_dim, dropout_p, context_dim=1):
        super(HybridModel, self).__init__()
        
        self.cnn_mode = config.Gan.CNN_MODE
        self.input_dim = input_dim
        self.context_dim = context_dim

        # --- Global Path (Transformer) ---
        self.transformer_encoder = build_transformer_encoder(
            input_dim=input_dim, # We strip the context feature before passing to encoder layer
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_p=dropout_p,
            context_dim=context_dim
        )
        
        # --- Local Path (Switchable CNN) ---
        cnn_output_dim = 128
        if self.cnn_mode == '1D':
            # Improved 1D CNN: Temporal Convolutional Network (TCN) style
            layers = []
            num_channels = [input_dim, 64, 64, cnn_output_dim]
            kernel_size = 3
            dropout = 0.2

            for i in range(len(num_channels) - 1):
                dilation_size = 2 ** i
                in_channels = num_channels[i]
                out_channels = num_channels[i+1]
                
                # Proper causal padding: (kernel_size - 1) * dilation for 'same' output length
                padding = (kernel_size - 1) * dilation_size
                
                layers += [
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=1, padding=padding, 
                              dilation=dilation_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            
            layers.append(nn.AdaptiveAvgPool1d(1))
            self.cnn_encoder = nn.Sequential(*layers)

        elif self.cnn_mode == '2D':
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(in_channels=64, out_channels=cnn_output_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_output_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Invalid CNN_MODE: {self.cnn_mode}. Choose '1D' or '2D'.")

        # --- Fusion & Generation ---
        # Note: GatedFusion outputs a vector of size d_model
        self.fusion_module = GatedFusion(d_model=d_model, cnn_dim=cnn_output_dim)
        
        self.decoder = build_gan_decoder(
            context_dim=d_model, # Output of fusion is d_model size
            noise_dim=noise_dim,
            output_dim=output_dim
        )
        self.noise_dim = noise_dim

    def forward(self, src, return_explainability=False):
        # src shape: (batch_size, seq_len, input_dim)
        # Use the centralized config for feature indices
        market_index_idx = config.Data.MARKET_INDEX_FEATURE_IDX
        context_dim = config.Data.CONTEXT_DIM
        
        # Safety check: if src has fewer features than expected, use fallback
        if src.shape[-1] <= market_index_idx + context_dim - 1:
             # Fallback: use last 2 features as context
             market_index_idx = max(0, src.shape[-1] - context_dim)
        
        # Extract context features: market_index_return + historical_similarity (2 features)
        context = src[:, :, market_index_idx:market_index_idx + context_dim]  # (batch, seq, context_dim)

        # 1. Global Path (Transformer) - with optional attention extraction
        if return_explainability:
            transformer_context, attention_weights = self.transformer_encoder(src, context=context, return_attention=True)
        else:
            transformer_context = self.transformer_encoder(src, context=context, return_attention=False)
            attention_weights = None
            
        last_step_transformer_context = transformer_context[:, -1, :] # Shape: (batch, d_model)

        # 2. Local Path (CNN)
        if self.cnn_mode == '1D':
            src_permuted = src.permute(0, 2, 1) # (batch, channels, seq_len)
            cnn_features = self.cnn_encoder(src_permuted)
        elif self.cnn_mode == '2D':
            # Accelerated GPU Implementation:
            # src is (B, L, C)
            image_tensor = batch_to_gaf_tensor(src, image_size=config.Data.IMAGE_SIZE) # -> (B, C, S, S)
            cnn_features = self.cnn_encoder(image_tensor)

        cnn_features = cnn_features.squeeze() # Remove pooled dimensions
        if cnn_features.dim() == 1:
             cnn_features = cnn_features.unsqueeze(0)

        # 3. Gated Fusion - with optional gate info
        if return_explainability:
            fused_context, gate_info = self.fusion_module(last_step_transformer_context, cnn_features, return_gate_info=True)
        else:
            fused_context = self.fusion_module(last_step_transformer_context, cnn_features)
            gate_info = None
        
        # 4. Conditional Generation (GAN)
        noise = torch.randn(src.size(0), self.noise_dim).to(config.Device.DEVICE)
        prediction = self.decoder(fused_context, noise)
        
        if return_explainability:
            return prediction, {
                'attention_weights': attention_weights,
                'gate_info': gate_info,
                'transformer_features': last_step_transformer_context.detach(),
                'cnn_features': cnn_features.detach(),
                'fused_context': fused_context.detach()
            }
        return prediction

def build_model(d_model, n_heads, n_layers, input_dim, noise_dim, output_dim, dropout_p) -> HybridModel:
    model = HybridModel(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        input_dim=input_dim,
        noise_dim=noise_dim,
        output_dim=output_dim,
        dropout_p=dropout_p,
        context_dim=config.Data.CONTEXT_DIM  # Use config: market_index + historical_similarity
    )
    model.to(config.Device.DEVICE)
    return model
