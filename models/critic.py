
import torch.nn as nn
from utils.config import config

class Critic(nn.Module):
    """
    The Critic (or Discriminator) model for the WGAN-GP.
    It takes a sequence of future price changes and outputs a single
    scalar score indicating how 'real' it thinks the sequence is.
    """
    def __init__(self, input_dim=None):
        super(Critic, self).__init__()
        # Use config if input_dim not provided
        input_dim = input_dim or config.Data.FUTURE_WINDOW_SIZE
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1) # Outputs a single scalar score
        )

    def forward(self, x):
        return self.model(x)

def build_critic(input_dim=None):
    """Builds and returns the Critic."""
    critic = Critic(input_dim=input_dim)
    critic.to(config.Device.DEVICE)
    return critic
