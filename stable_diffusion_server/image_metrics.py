"""Image comparison metrics for validating CuteZImage output quality.

Provides pixel-level and structural comparison without heavy ML dependencies.
For FID scores, use the clean-fid package separately on larger image sets.

Metrics:
- Pixel MAE/MSE: Mean absolute/squared error per pixel channel
- PSNR: Peak signal-to-noise ratio (higher = more similar)
- SSIM: Structural similarity index (higher = more similar, 1.0 = identical)
- Max pixel error: Worst-case pixel difference
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pixel_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean absolute error per pixel (0-255 scale)."""
    return (img1.float() - img2.float()).abs().mean().item()


def pixel_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Mean squared error per pixel."""
    return (img1.float() - img2.float()).pow(2).mean().item()


def max_pixel_error(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Maximum absolute pixel difference."""
    return (img1.float() - img2.float()).abs().max().item()


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher = better. Inf = identical."""
    mse = pixel_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(max_val * max_val / mse)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 6.5025,  # (0.01 * 255)^2
    C2: float = 58.5225,  # (0.03 * 255)^2
) -> float:
    """Structural Similarity Index (SSIM).

    Computes SSIM between two images using a Gaussian window.
    Images should be (C, H, W) or (H, W, C) uint8/float tensors.

    Returns a scalar in [0, 1] where 1.0 = identical.
    """
    # Ensure (1, C, H, W) float
    x = img1.float()
    y = img2.float()
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        x = x.permute(2, 0, 1)
        y = y.permute(2, 0, 1)
    if x.ndim == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    channels = x.shape[1]

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor([
        math.exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2))
        for i in range(window_size)
    ], dtype=torch.float32, device=x.device)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    padding = window_size // 2

    mu1 = F.conv2d(x, window, padding=padding, groups=channels)
    mu2 = F.conv2d(y, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=padding, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def compare_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> dict:
    """Full comparison of two images. Returns dict of all metrics.

    Args:
        img1, img2: Images as tensors (H, W, C) or (C, H, W), uint8 or float.

    Returns:
        Dict with mae, mse, psnr, ssim, max_pixel_error keys.
    """
    return {
        "mae": pixel_mae(img1, img2),
        "mse": pixel_mse(img1, img2),
        "psnr_db": psnr(img1, img2),
        "ssim": ssim(img1, img2),
        "max_pixel_error": max_pixel_error(img1, img2),
        "images_identical": torch.equal(img1, img2),
    }


def pil_to_tensor(img) -> torch.Tensor:
    """Convert a PIL Image to a (H, W, C) uint8 tensor."""
    import numpy as np
    return torch.from_numpy(np.array(img))
