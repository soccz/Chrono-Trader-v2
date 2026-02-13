"""
Model Explainability Analyzer
Provides tools for analyzing and visualizing model decisions.

Features:
1. Attention Map Extraction & Visualization
2. Prototype Bank Analysis
3. GAN Multi-noise Sampling for Uncertainty
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
from datetime import datetime

from utils.config import config
from utils.logger import logger


class ExplainabilityAnalyzer:
    """
    Analyzes and visualizes model explainability features.
    """
    
    def __init__(self, output_dir="static/explainability"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # 1. Attention Map Analysis
    # ==========================================
    
    def visualize_attention(self, attention_weights, market_name="", save=True):
        """
        Visualizes attention weights from Transformer layers.
        
        Args:
            attention_weights: List of attention tensors from each layer
                              Each tensor: (batch, n_heads, seq_len, seq_len)
            market_name: Name of the market for labeling
            save: Whether to save the figure
            
        Returns:
            fig: matplotlib figure object
        """
        if attention_weights is None or len(attention_weights) == 0:
            logger.warning("No attention weights provided")
            return None
        
        n_layers = len(attention_weights)
        
        # Create figure with subplots for each layer
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
        if n_layers == 1:
            axes = [axes]
        
        for i, attn in enumerate(attention_weights):
            # attn: (batch, n_heads, seq_len, seq_len) or (batch, seq_len, seq_len)
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu().numpy()
            
            # Average across batch and heads
            if attn.ndim == 4:
                attn_avg = attn.mean(axis=(0, 1))  # (seq_len, seq_len)
            elif attn.ndim == 3:
                attn_avg = attn.mean(axis=0)
            else:
                attn_avg = attn
            
            # Plot heatmap
            im = axes[i].imshow(attn_avg, cmap='viridis', aspect='auto')
            axes[i].set_title(f'Layer {i+1}')
            axes[i].set_xlabel('Key Position (Time)')
            axes[i].set_ylabel('Query Position (Time)')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        fig.suptitle(f'Attention Maps - {market_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'attention_{market_name}_{timestamp}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
            # Save latest for web display
            latest_path = os.path.join(self.output_dir, 'latest_attention.png')
            fig.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Attention map saved to {filepath} and {latest_path}")
        
        return fig
    
    def get_temporal_importance(self, attention_weights):
        """
        Calculates which time steps are most important for the prediction.
        
        Args:
            attention_weights: List of attention tensors
            
        Returns:
            importance: (seq_len,) array of importance scores
        """
        if attention_weights is None or len(attention_weights) == 0:
            return None
        
        # Use the last layer's attention (closest to output)
        last_attn = attention_weights[-1]
        if isinstance(last_attn, torch.Tensor):
            last_attn = last_attn.detach().cpu().numpy()
        
        # Average across batch and heads
        if last_attn.ndim == 4:
            last_attn = last_attn.mean(axis=(0, 1))
        elif last_attn.ndim == 3:
            last_attn = last_attn.mean(axis=0)
        
        # The last row shows what the final position attends to
        # This indicates which past time steps influenced the prediction
        importance = last_attn[-1, :]  # (seq_len,)
        
        return importance
    
    # ==========================================
    # 2. Prototype Bank Analysis
    # ==========================================
    
    def visualize_prototypes(self, gate_info, save=True):
        """
        Visualizes the learned prototype bank using t-SNE.
        
        Args:
            gate_info: Dictionary containing 'prototype_bank' array
            save: Whether to save the figure
            
        Returns:
            fig: matplotlib figure object
        """
        if gate_info is None or 'prototype_bank' not in gate_info:
            logger.warning("No prototype bank in gate_info")
            return None
        
        prototypes = gate_info['prototype_bank']  # (n_prototypes, d_model)
        n_prototypes = prototypes.shape[0]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. t-SNE visualization
        if n_prototypes > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, n_prototypes-1))
            prototypes_2d = tsne.fit_transform(prototypes)
            
            # Cluster the prototypes
            n_clusters = min(4, n_prototypes)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(prototypes)
            
            scatter = axes[0].scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], 
                                     c=clusters, cmap='tab10', s=100)
            for i in range(n_prototypes):
                axes[0].annotate(f'P{i}', (prototypes_2d[i, 0], prototypes_2d[i, 1]))
            axes[0].set_title('Prototype Clusters (t-SNE)')
            axes[0].set_xlabel('Dimension 1')
            axes[0].set_ylabel('Dimension 2')
        
        # 2. Prototype similarity matrix
        prototype_norms = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(prototype_norms, prototype_norms.T)
        
        sns.heatmap(sim_matrix, ax=axes[1], cmap='coolwarm', center=0,
                   xticklabels=[f'P{i}' for i in range(n_prototypes)],
                   yticklabels=[f'P{i}' for i in range(n_prototypes)])
        axes[1].set_title('Prototype Similarity Matrix')
        
        # 3. Prototype activation distribution (if available)
        if 'transformer_proto_sims' in gate_info:
            trans_sims = gate_info['transformer_proto_sims'].flatten()
            cnn_sims = gate_info['cnn_proto_sims'].flatten()
            
            x = np.arange(n_prototypes)
            width = 0.35
            axes[2].bar(x - width/2, trans_sims[:n_prototypes], width, label='Transformer', alpha=0.8)
            axes[2].bar(x + width/2, cnn_sims[:n_prototypes], width, label='CNN', alpha=0.8)
            axes[2].set_xlabel('Prototype Index')
            axes[2].set_ylabel('Similarity Score')
            axes[2].set_title('Prototype Activations')
            axes[2].legend()
            axes[2].set_xticks(x)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'prototypes_{timestamp}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
            # Save latest for web display
            latest_path = os.path.join(self.output_dir, 'latest_prototypes.png')
            fig.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Prototype visualization saved to {filepath} and {latest_path}")
        
        return fig
    
    def get_gate_explanation(self, gate_info):
        """
        Provides human-readable explanation of the gating decision.
        
        Args:
            gate_info: Dictionary from ExplainableGatedFusion
            
        Returns:
            explanation: Dictionary with interpretation
        """
        if gate_info is None:
            return {"error": "No gate info provided"}
        
        trans_sim = float(gate_info.get('transformer_similarity', [[0]])[0][0])
        cnn_sim = float(gate_info.get('cnn_similarity', [[0]])[0][0])
        gate_val = float(gate_info.get('gate_values', [0.5])[0])
        
        dominant_path = 'Transformer' if gate_val > 0.5 else 'CNN'
        explanation = {
            'transformer_similarity': trans_sim,
            'cnn_similarity': cnn_sim,
            'gate_value': gate_val,
            'dominant_path': dominant_path,
            'interpretation': ''
        }

        # Primary explanation follows the actual gate decision.
        if dominant_path == 'Transformer':
            if trans_sim >= cnn_sim:
                explanation['interpretation'] = "Gate selected Transformer and similarity also favors global trend structure."
            else:
                explanation['interpretation'] = "Gate selected Transformer despite stronger local similarity; global context dominated."
        else:
            if cnn_sim >= trans_sim:
                explanation['interpretation'] = "Gate selected CNN and similarity also favors local pattern signals."
            else:
                explanation['interpretation'] = "Gate selected CNN despite stronger global similarity; local signal dominated."
        
        return explanation
    
    # ==========================================
    # 3. GAN Multi-Noise Sampling
    # ==========================================
    
    def sample_predictions(self, model, sequence_tensor, n_samples=20):
        """
        Generates multiple predictions with different noise vectors.
        
        Args:
            model: HybridModel instance
            sequence_tensor: Input sequence (batch, seq_len, features)
            n_samples: Number of noise samples
            
        Returns:
            predictions: (n_samples, output_dim) array
            mean_pred: (output_dim,) mean prediction
            std_pred: (output_dim,) standard deviation
            uncertainty: scalar uncertainty score
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Get fused context once (deterministic part)
            # We need to manually extract the fused context
            market_index_idx = config.Data.MARKET_INDEX_FEATURE_IDX
            context_dim = config.Data.CONTEXT_DIM
            context = sequence_tensor[:, :, market_index_idx:market_index_idx + context_dim]
            
            transformer_context = model.transformer_encoder(sequence_tensor, context=context)
            last_step = transformer_context[:, -1, :]
            
            if model.cnn_mode == '1D':
                src_permuted = sequence_tensor.permute(0, 2, 1)
                cnn_features = model.cnn_encoder(src_permuted).squeeze()
            else:
                from utils.image_converter import batch_to_gaf_tensor
                image_tensor = batch_to_gaf_tensor(sequence_tensor, image_size=config.Data.IMAGE_SIZE)
                cnn_features = model.cnn_encoder(image_tensor).squeeze()
            
            if cnn_features.dim() == 1:
                cnn_features = cnn_features.unsqueeze(0)
            
            fused_context = model.fusion_module(last_step, cnn_features)
            
            # Sample with different noise vectors
            for _ in range(n_samples):
                noise = torch.randn(sequence_tensor.size(0), model.noise_dim).to(config.Device.DEVICE)
                pred = model.decoder(fused_context, noise)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions).squeeze()  # (n_samples, output_dim)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        uncertainty = np.sum(std_pred)  # Total uncertainty
        
        return {
            'predictions': predictions,
            'mean': mean_pred,
            'std': std_pred,
            'uncertainty': uncertainty,
            'confidence_interval_95': (mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred)
        }
    
    def visualize_prediction_distribution(self, sample_result, market_name="", save=True):
        """
        Visualizes the distribution of predictions from multi-noise sampling.
        
        Args:
            sample_result: Output from sample_predictions()
            market_name: Name of market for labeling
            save: Whether to save the figure
            
        Returns:
            fig: matplotlib figure object
        """
        predictions = sample_result['predictions']
        mean = sample_result['mean']
        std = sample_result['std']
        ci = sample_result['confidence_interval_95']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Prediction paths
        x = np.arange(len(mean))
        for i, pred in enumerate(predictions):
            axes[0].plot(x, pred, alpha=0.3, color='blue', linewidth=0.5)
        axes[0].plot(x, mean, color='red', linewidth=2, label='Mean Prediction')
        axes[0].fill_between(x, ci[0], ci[1], alpha=0.2, color='red', label='95% CI')
        axes[0].set_xlabel('Future Time Step')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title(f'Prediction Distribution - {market_name}')
        axes[0].legend()
        
        # 2. Uncertainty by time step
        axes[1].bar(x, std, color='orange', alpha=0.7)
        axes[1].set_xlabel('Future Time Step')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].set_title(f'Uncertainty by Time Step (Total: {sample_result["uncertainty"]:.4f})')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'prediction_dist_{market_name}_{timestamp}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
            # Save latest for web display
            latest_path = os.path.join(self.output_dir, 'latest_prediction_dist.png')
            fig.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            logger.info(f"Prediction distribution saved to {filepath} and {latest_path}")
        
        return fig


# Singleton instance
explainability_analyzer = ExplainabilityAnalyzer()


def analyze_prediction(model, sequence_tensor, market_name=""):
    """
    Convenience function to run full explainability analysis on a prediction.
    
    Returns:
        result: Dictionary with prediction, explainability info, and visualizations
    """
    analyzer = explainability_analyzer
    
    # Get prediction with explainability info
    with torch.no_grad():
        prediction, explain_info = model(sequence_tensor, return_explainability=True)
    
    result = {
        'prediction': prediction.cpu().numpy(),
        'attention_weights': explain_info['attention_weights'],
        'gate_info': explain_info['gate_info'],
    }
    
    # Analyze temporal importance
    importance = analyzer.get_temporal_importance(explain_info['attention_weights'])
    result['temporal_importance'] = importance
    
    # Get gate explanation
    result['gate_explanation'] = analyzer.get_gate_explanation(explain_info['gate_info'])
    
    # Multi-noise sampling
    sample_result = analyzer.sample_predictions(model, sequence_tensor, n_samples=20)
    result['multi_sample'] = sample_result
    
    # Generate visualizations
    result['attention_fig'] = analyzer.visualize_attention(
        explain_info['attention_weights'], market_name=market_name
    )
    result['prototype_fig'] = analyzer.visualize_prototypes(
        explain_info['gate_info']
    )
    result['distribution_fig'] = analyzer.visualize_prediction_distribution(
        sample_result, market_name=market_name
    )
    
    return result
