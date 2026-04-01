
import torch
import torch.nn as nn
import numpy as np
from utils.config import config
from utils.image_converter import to_gaf_image, batch_to_gaf_tensor
from models.transformer_encoder import build_transformer_encoder
from models.gan_decoder import build_gan_decoder
from models.cvae_decoder import build_cvae_decoder

class FiLMRegimeConditioning(nn.Module):
    """Feature-wise Linear Modulation: scale/shift hidden state by BTC regime."""
    def __init__(self, n_regimes: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(n_regimes, hidden_dim)
        self.scale = nn.Linear(hidden_dim, hidden_dim)
        self.shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden_dim), regime: (B,) long
        e = self.embed(regime)
        return h * self.scale(e) + self.shift(e)


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
        
        # --- Gate Regularization (ENHANCED) ---
        # 1. Keep mean in hybrid range (0.3 - 0.7)
        gate_mean = gate.mean()
        gate_range_reg = torch.relu(0.3 - gate_mean) + torch.relu(gate_mean - 0.7)
        
        # 2. Encourage diversity within batch (prevent collapse to single value)
        gate_per_sample = gate.mean(dim=1)  # (batch,)
        gate_diversity = torch.std(gate_per_sample, unbiased=False)
        gate_diversity_loss = -0.1 * gate_diversity  # negative → maximise diversity
        
        # 3. Prototype diversity - penalise high inter-prototype similarity
        proto_norms = nn.functional.normalize(self.prototype_bank, p=2, dim=1)
        proto_sim_matrix = torch.mm(proto_norms, proto_norms.t())
        mask = ~torch.eye(proto_sim_matrix.size(0), dtype=torch.bool, device=proto_sim_matrix.device)
        proto_diversity_loss = 0.01 * proto_sim_matrix[mask].mean()  # positive → penalise similarity
        
        # Combined: gate_range_reg and proto_diversity_loss are penalties (+),
        # gate_diversity_loss is a reward (−) → all added together is correct.
        total_reg_loss = gate_range_reg + gate_diversity_loss + proto_diversity_loss
        self._gate_reg_loss = total_reg_loss
        self._gate_diversity = gate_diversity.item()
        self._proto_diversity_loss = proto_diversity_loss.item()
        
        # Weighted fusion
        fused = gate * global_features + (1 - gate) * local_proj
        
        if return_gate_info:
            gate_info = {
                'transformer_similarity': transformer_sim.detach().cpu().numpy(),
                'cnn_similarity': cnn_sim.detach().cpu().numpy(),
                'transformer_proto_sims': transformer_proto_sims.detach().cpu().numpy(),
                'cnn_proto_sims': cnn_proto_sims.detach().cpu().numpy(),
                'gate_values': gate.mean(dim=1).detach().cpu().numpy(),  # Average gate for interpretability
                'gate_diversity': gate_diversity.item(),
                'prototype_bank': self.prototype_bank.detach().cpu().numpy(),
                'gate_reg_loss': total_reg_loss.item(),
                'proto_diversity_loss': proto_diversity_loss.item()
            }
            return fused, gate_info
        
        return fused



# Keep old class name as alias for backward compatibility
GatedFusion = ExplainableGatedFusion


class CausalConv1d(nn.Module):
    """
    Left-padded causal convolution.

    This keeps the temporal dimension aligned while preventing any look-ahead
    leakage from the local branch.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    """Residual TCN block with two causal convolutions."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        return self.activation(out + residual)


class CausalResidualTCN(nn.Module):
    """
    Compact causal TCN encoder for the local branch.

    Output shape: (B, output_dim, 1)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels=(64, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for depth, out_channels in enumerate(hidden_channels):
            dilation = 2 ** depth
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(in_channels, output_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.projection(x)
        x = self.pool(x)
        return x


class HybridModel(nn.Module):
    """
    Hybrid Model: Transformer (global) + attention-guided TCN (local) + gated fusion.

    decoder_mode:
      'gan'  - original GAN decoder (backward compat)
      'cvae' - CVAE decoder for probabilistic forecasting (H3, paper main)

    pe_mode: passed through to TransformerEncoder / CyclePE
      'static' | 'concat_a' | 'cycle_pe' | 'cycle_pe_full'
    """
    def __init__(self, d_model, n_heads, n_layers, input_dim, noise_dim, output_dim,
                 dropout_p, context_dim=1, use_causal_mask=None,
                 decoder_mode='cvae', pe_mode='cycle_pe_full'):
        super().__init__()

        self.cnn_mode     = config.Gan.CNN_MODE
        self.input_dim    = input_dim
        self.context_dim  = context_dim
        self.decoder_mode = decoder_mode
        self.pe_mode      = pe_mode
        if use_causal_mask is None:
            use_causal_mask = getattr(config.Gan, 'USE_CAUSAL_MASK', True)

        # --- Global Path (Transformer + CyclePE) ---
        self.transformer_encoder = build_transformer_encoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_p=dropout_p,
            context_dim=context_dim,
            use_causal_mask=use_causal_mask,
            pe_mode=pe_mode,
        )
        
        # --- Local Path (Switchable CNN) ---
        cnn_output_dim = 128
        if self.cnn_mode == '1D':
            # Causal residual TCN: no look-ahead, explicit skip connections.
            self.cnn_encoder = CausalResidualTCN(
                input_dim=input_dim,
                output_dim=cnn_output_dim,
                hidden_channels=(64, 64, 128),
                kernel_size=3,
                dropout=0.2,
            )

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

        # --- Fusion & Regime Conditioning ---
        self.fusion_module = GatedFusion(d_model=d_model, cnn_dim=cnn_output_dim)
        self.film = FiLMRegimeConditioning(n_regimes=4, hidden_dim=d_model)

        # --- Decoder (GAN or CVAE) ---
        self.noise_dim = noise_dim
        if decoder_mode == 'cvae':
            self.decoder = build_cvae_decoder(
                context_dim=d_model, output_dim=output_dim, dropout_p=dropout_p
            )
        else:
            self.decoder = build_gan_decoder(
                context_dim=d_model, noise_dim=noise_dim,
                output_dim=output_dim, dropout_p=dropout_p
            )

    def _mask_context_channels(self, src: torch.Tensor) -> torch.Tensor:
        """
        Remove raw context channels for non-concat modes.

        In concat_a, the context is intentionally part of the raw input.
        In other modes, the context should enter through PE/context paths only.
        """
        pe_mode = getattr(self, "pe_mode", getattr(config.Gan, "PE_MODE", "cycle_pe_full"))
        context_dim = int(getattr(self, "context_dim", getattr(config.Data, "CONTEXT_DIM", 0)) or 0)

        if pe_mode == 'concat_a' or context_dim <= 0:
            return src

        context_start = config.Data.MARKET_INDEX_FEATURE_IDX
        context_end = context_start + context_dim
        if src.shape[-1] < context_end:
            return src

        masked = src.clone()
        masked[:, :, context_start:context_end] = 0.0
        return masked

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
        raw_src = self._mask_context_channels(src)

        # 1. Global Path (Transformer) - always extract attention for CNN guidance
        transformer_context, attention_weights_list = self.transformer_encoder(
            raw_src, context=context, return_attention=True
        )
        last_step_transformer_context = transformer_context[:, -1, :]  # (B, d_model)

        # Attention-guided CNN: soft masking with last token's attention row
        # attention_weights_list[-1]: (B, seq, seq) → last token row → (B, seq)
        last_attn = attention_weights_list[-1]  # (B, seq, seq)
        importance = torch.softmax(last_attn[:, -1, :], dim=-1)  # (B, seq)
        guided_src = raw_src * importance.unsqueeze(-1)  # (B, seq, input_dim)

        # 2. Local Path (CNN) on attention-guided input
        if self.cnn_mode == '1D':
            src_permuted = guided_src.permute(0, 2, 1)  # (B, C, seq)
            cnn_features = self.cnn_encoder(src_permuted)
        elif self.cnn_mode == '2D':
            image_tensor = batch_to_gaf_tensor(guided_src, image_size=config.Data.IMAGE_SIZE)
            cnn_features = self.cnn_encoder(image_tensor)

        # Keep a stable shape for all batch sizes: (B, cnn_output_dim)
        cnn_features = cnn_features.flatten(start_dim=1)

        # 3. Gated Fusion
        if return_explainability:
            fused_context, gate_info = self.fusion_module(
                last_step_transformer_context, cnn_features, return_gate_info=True
            )
        else:
            fused_context = self.fusion_module(last_step_transformer_context, cnn_features)
            gate_info = None

        # 4. FiLM regime conditioning
        # btc_regime is the last feature before factor interactions; extract from last timestep
        regime_idx = config.Data.FEATURE_COLUMNS.index('btc_regime') if 'btc_regime' in config.Data.FEATURE_COLUMNS else None
        if regime_idx is not None and src.shape[-1] > regime_idx:
            regime_values = src[:, -1, regime_idx]
            if torch.all((regime_values >= -1e-6) & (regime_values <= 1.0 + 1e-6)):
                regime = torch.round(regime_values * 3.0).long().clamp(0, 3)
            else:
                regime = torch.round(regime_values).long().clamp(0, 3)
        else:
            regime = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
        film = getattr(self, "film", None)
        if film is not None:
            fused_context = film(fused_context, regime)
        
        # Propagate gate regularization loss to top-level model
        # so trainer.py's getattr(generator, '_gate_reg_loss') can find it
        self._gate_reg_loss = getattr(self.fusion_module, '_gate_reg_loss', torch.tensor(0.0).to(config.Device.DEVICE))
        
        # 4. Conditional Generation
        decoder_mode = getattr(self, "decoder_mode", "gan")
        if decoder_mode == 'cvae':
            self._cvae_fused_context = fused_context
            mu, log_sigma = self.decoder(fused_context)
            prediction = mu  # main prediction is the mean
            self._cvae_mu = mu
            self._cvae_log_sigma = log_sigma
            self._cvae_kl_loss = torch.tensor(0.0, device=fused_context.device)  # no KL
        else:
            noise_dim = int(getattr(self, "noise_dim", getattr(config.Gan, "NOISE_DIM", 100)) or 100)
            noise = torch.randn(src.size(0), noise_dim, device=fused_context.device)
            prediction = self.decoder(fused_context, noise)

        if return_explainability:
            return prediction, {
                'attention_weights': attention_weights_list,
                'attention_importance': importance.detach(),  # (B, seq) — for heatmap
                'gate_info': gate_info,
                'transformer_features': last_step_transformer_context.detach(),
                'cnn_features': cnn_features.detach(),
                'fused_context': fused_context.detach(),
                'regime': regime.detach(),
            }
        return prediction

    def forward_train(self, src: torch.Tensor, y_true: torch.Tensor):
        """Heteroscedastic training: y_true stored for NLL loss in trainer."""
        self._y_true_for_cvae = y_true  # keep attribute name for trainer compat
        out = self.forward(src)
        self._y_true_for_cvae = None
        return out

    @torch.no_grad()
    def predict_interval(self, src: torch.Tensor,
                         n_samples: int = 100, alpha: float = 0.80):
        """
        CVAE-only. Returns (mean, pi_low, pi_high) each (B, output_dim).
        pi_low  = (1-alpha)/2 quantile across MC samples.
        """
        assert getattr(self, "decoder_mode", "gan") == 'cvae', "predict_interval requires decoder_mode='cvae'"
        self.eval()
        _, expl = self.forward(src, return_explainability=True)
        fused = expl['fused_context']
        return self.decoder.predict_interval(fused, n_samples=n_samples, alpha=alpha)

def build_model(d_model, n_heads, n_layers, input_dim, noise_dim, output_dim, dropout_p,
                decoder_mode='cvae', pe_mode='cycle_pe_full') -> HybridModel:
    model = HybridModel(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        input_dim=input_dim, noise_dim=noise_dim, output_dim=output_dim,
        dropout_p=dropout_p,
        context_dim=config.Data.CONTEXT_DIM,
        use_causal_mask=getattr(config.Gan, 'USE_CAUSAL_MASK', True),
        decoder_mode=decoder_mode,
        pe_mode=pe_mode,
    )
    model.to(config.Device.DEVICE)
    return model
