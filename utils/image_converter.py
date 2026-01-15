import numpy as np
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler

def to_gaf_image(time_series: np.ndarray, image_size: int):
    """
    Converts a 1D time-series into a 2D Gramian Angular Field (GAF) image.

    Args:
        time_series (np.ndarray): A 1D numpy array representing the time-series.
        image_size (int): The desired output image size (height and width).

    Returns:
        np.ndarray: A 2D GAF image of shape (image_size, image_size).
    """
    # Ensure the time series is a 1D array
    if time_series.ndim > 1:
        time_series = time_series.flatten()

    # Scale the time series to the range [-1, 1] as required by GAF
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ts_scaled = scaler.fit_transform(time_series.reshape(1, -1))

    # Resample the time series to match the desired image size
    # This is a simple linear interpolation. More sophisticated methods could be used.
    ts_resampled = np.interp(
        np.linspace(0, len(ts_scaled[0]) - 1, image_size),
        np.arange(len(ts_scaled[0])),
        ts_scaled[0]
    ).reshape(1, -1)

    # Create a GAF transformer and apply it
    gaf = GramianAngularField(image_size=image_size, method='summation')
    gaf_image = gaf.fit_transform(ts_resampled)

    # The output is (1, image_size, image_size), so we squeeze it to (image_size, image_size)
    return gaf_image.squeeze(0)

import torch

def batch_to_gaf_tensor(src_tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Vectorized GPU implementation of GASF (Gramian Angular Summation Field).
    
    Args:
        src_tensor (torch.Tensor): Shape (Batch, Seq_Len, Features)
        image_size (int): Output width/height
        
    Returns:
        torch.Tensor: Shape (Batch, Features, image_size, image_size)
    """
    # src: (B, L, C) -> Permute to (B, C, L) for interpolation
    x = src_tensor.permute(0, 2, 1) # (B, C, L)
    
    # 1. Resample/Interpolate to image_size
    # Use 'linear' mode for 1D signal
    if x.size(2) != image_size:
        x = torch.nn.functional.interpolate(x, size=image_size, mode='linear', align_corners=False)
    # x is now (B, C, S) where S = image_size
    
    # 2. MinMax Scale to [-1, 1] per channel per sample
    # We want to normalize along the time dimension (dim=2)
    min_val = x.min(dim=2, keepdim=True)[0]
    max_val = x.max(dim=2, keepdim=True)[0]
    
    # Avoid division by zero
    diff = max_val - min_val
    diff[diff == 0] = 1.0
    
    # Scale to [0, 1] then to [-1, 1]
    # x_std = (x - min) / (max - min)
    # x_scaled = x_std * 2 - 1
    x_scaled = ((x - min_val) / diff) * 2 - 1
    
    # Clamp to ensure numerical stability for sqrt (1 - x^2)
    x_scaled = torch.clamp(x_scaled, -0.999999, 0.999999)
    
    # 3. Compute GASF
    # Formula: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
    # where a = arccos(x_i), b = arccos(x_j)
    # cos(arccos(x)) = x
    # sin(arccos(x)) = sqrt(1 - x^2)
    # So GASF_ij = x_i * x_j - sqrt(1 - x_i^2) * sqrt(1 - x_j^2)
    
    # Prepare broadcasting
    # x_i: (B, C, S, 1)
    # x_j: (B, C, 1, S)
    x_i = x_scaled.unsqueeze(-1)
    x_j = x_scaled.unsqueeze(-2)
    
    sqrt_term = torch.sqrt(1 - x_scaled**2)
    sqrt_i = sqrt_term.unsqueeze(-1)
    sqrt_j = sqrt_term.unsqueeze(-2)
    
    # Result: (B, C, S, S)
    gaf_tensor = (x_i * x_j) - (sqrt_i * sqrt_j)
    
    return gaf_tensor
