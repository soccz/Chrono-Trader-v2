import torch
import torch.nn as nn

class GANDecoder(nn.Module):
    """
    Enhanced Generator model for GAN with deeper architecture.
    Includes dropout for regularization and an additional hidden layer.
    """
    def __init__(self, context_dim, noise_dim, output_dim, dropout_p=0.1):
        super(GANDecoder, self).__init__()
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            # Layer 1: Input
            nn.Linear(context_dim + noise_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_p),
            
            # Layer 2: Expansion
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_p),
            
            # Layer 3: Deep feature extraction (NEW)
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            # Layer 4: Output
            nn.Linear(256, output_dim),
        )

    def forward(self, context_vector, noise):
        combined_input = torch.cat((context_vector, noise), dim=1)
        output = self.model(combined_input)
        return output

def build_gan_decoder(context_dim, noise_dim, output_dim, dropout_p=0.1):
    """Builds and returns the enhanced GAN decoder."""
    return GANDecoder(
        context_dim=context_dim,
        noise_dim=noise_dim,
        output_dim=output_dim,
        dropout_p=dropout_p
    )
