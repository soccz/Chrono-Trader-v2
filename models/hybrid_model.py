import torch
import torch.nn as nn
from utils.config import config
from utils.image_converter import to_gaf_image
from models.transformer_encoder import build_transformer_encoder
from models.gan_decoder import build_gan_decoder

class HybridModel(nn.Module):
    """
    A novel hybrid model for time-series prediction, featuring a switchable
    CNN architecture (1D or 2D) running in parallel with a Transformer.

    This model allows for experimenting with two different approaches for capturing
    local patterns, which are then fused with global context from the Transformer.

    Architecture:
    1.  **Parallel Feature Extraction**:
        a) **Global Path (Transformer)**: Captures long-range dependencies.
        b) **Local Path (Switchable CNN)**: Based on `config.CNN_MODE`:
            - **'1D' Mode**: A 1D CNN extracts local temporal motifs directly
              from the 1D time-series data.
            - **'2D' Mode**: The 1D time-series is first converted into a 2D GAF
              image. A 2D CNN then extracts spatial patterns from this image,
              which represent complex temporal correlations.

    2.  **Feature Fusion**: The global context from the Transformer and the local
        features from the selected CNN path are concatenated.

    3.  **Conditional Generation (GAN Decoder)**: A GAN-based Decoder generates
        the final prediction, conditioned on the rich, fused feature vector.
    """
    def __init__(self, d_model, n_heads, n_layers, input_dim, noise_dim, output_dim, dropout_p):
        super(HybridModel, self).__init__()
        
        self.cnn_mode = config.CNN_MODE
        self.input_dim = input_dim

        # --- Global Path (Transformer) ---
        self.transformer_encoder = build_transformer_encoder(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_p=dropout_p
        )
        
        # --- Local Path (Switchable CNN) ---
        cnn_output_dim = 128
        if self.cnn_mode == '1D':
            self.cnn_encoder = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=cnn_output_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        elif self.cnn_mode == '2D':
            # For 2D, we process each feature as a separate channel in the image
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=cnn_output_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Invalid CNN_MODE: {self.cnn_mode}. Choose '1D' or '2D'.")

        # --- Fusion & Generation ---
        fused_dim = d_model + cnn_output_dim
        self.decoder = build_gan_decoder(
            context_dim=fused_dim,
            noise_dim=noise_dim,
            output_dim=output_dim
        )
        self.noise_dim = noise_dim

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        
        # 1. Global Path (Transformer)
        transformer_context = self.transformer_encoder(src)
        last_step_transformer_context = transformer_context[:, -1, :] # Shape: (batch_size, d_model)

        # 2. Local Path (CNN)
        if self.cnn_mode == '1D':
            src_permuted = src.permute(0, 2, 1) # (batch, channels, seq_len)
            cnn_features = self.cnn_encoder(src_permuted)
        elif self.cnn_mode == '2D':
            # Convert each time-series feature in the batch to a GAF image
            # This is computationally intensive and done on the CPU for simplicity here.
            # For performance, this could be batched and done on the GPU.
            batch_images = []
            for i in range(src.size(0)):
                # Create a multi-channel image, one channel per feature
                channels = [to_gaf_image(src[i, :, j].cpu().numpy(), config.IMAGE_SIZE) for j in range(self.input_dim)]
                batch_images.append(np.stack(channels, axis=0))
            
            image_tensor = torch.FloatTensor(np.array(batch_images)).to(config.DEVICE)
            cnn_features = self.cnn_encoder(image_tensor)

        cnn_features = cnn_features.squeeze() # Remove pooled dimensions
        if cnn_features.dim() == 1:
             cnn_features = cnn_features.unsqueeze(0) # Handle batch size of 1

        # 3. Feature Fusion
        fused_context = torch.cat((last_step_transformer_context, cnn_features), dim=1)
        
        # 4. Conditional Generation (GAN)
        noise = torch.randn(src.size(0), self.noise_dim).to(config.DEVICE)
        prediction = self.decoder(fused_context, noise)
        
        return prediction

def build_model(d_model, n_heads, n_layers, input_dim, noise_dim, output_dim, dropout_p) -> HybridModel:
    model = HybridModel(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        input_dim=input_dim,
        noise_dim=noise_dim,
        output_dim=output_dim,
        dropout_p=dropout_p
    )
    model.to(config.DEVICE)
    return model
