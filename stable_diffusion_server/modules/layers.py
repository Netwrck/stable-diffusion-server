import math
from typing import Literal

import torch
from torch import Tensor, nn
from einops import rearrange
from torch.nn.functional import silu

from .attention import Attention


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        return self.base_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = SamePadConv2d(in_channels, self.out_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.conv2 = SamePadConv2d(self.out_channels, self.out_channels, kernel_size=3, bias=False)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv2d(in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = Attention(heads=num_heads, dim=channels)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = x.reshape(b, c, h * w)
        h_ = self.norm(h_)
        qkv = self.qkv(h_)
        h_ = self.attention(qkv)
        h_ = self.proj_out(h_)
        h_ = h_.reshape(b, c, h, w)
        return x + h_


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        An (N, D) Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Embeds a tensor of indices.

        Args:
            x: Tensor of shape (..., n_axes) and type int64.

        Returns:
            Tensor of shape (..., D) and type float32.
        """
        if x.shape[-1] != len(self.axes_dim):
            raise ValueError(f"Expected {len(self.axes_dim)} axes, got {x.shape[-1]}")
        # these are the positions of the different axes in the final embedding
        splits = [0] + list(torch.cumsum(torch.tensor(self.axes_dim), 0))
        # this is a one-hot encoding of the axes, but more general
        embeddings = [
            timestep_embedding(x[..., i], self.axes_dim[i], max_period=self.theta) for i in range(len(self.axes_dim))
        ]
        return torch.cat(embeddings, dim=-1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        x = self.act(x)
        x = self.out_proj(x)
        return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias

        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=True,
            context_dim=hidden_size,
        )
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_img = self.mlp_block(hidden_size, mlp_ratio)
        self.mlp_txt = self.mlp_block(hidden_size, mlp_ratio)
        self.ada_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def mlp_block(self, hidden_size: int, mlp_ratio: float) -> nn.Module:
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_norm(vec).chunk(6, dim=1)
        pe_img, pe_txt = pe[:, txt.shape[1] :, ...], pe[:, : txt.shape[1], ...]

        # attention
        img_res = img
        txt_res = txt
        img = self.norm1_img(img)
        txt = self.norm1_txt(txt)
        img = (1 + scale_msa).unsqueeze(1) * img + shift_msa.unsqueeze(1)
        txt = (1 + scale_msa).unsqueeze(1) * txt + shift_msa.unsqueeze(1)
        img = self.attn(img, context=txt, pos_embed=pe)
        img = gate_msa.unsqueeze(1) * img
        img = img_res + img

        # mlp
        img_res = img
        txt_res = txt
        img = self.norm2_img(img)
        txt = self.norm2_txt(txt)
        img = (1 + scale_mlp).unsqueeze(1) * img + shift_mlp.unsqueeze(1)
        txt = (1 + scale_mlp).unsqueeze(1) * txt + shift_mlp.unsqueeze(1)
        img = self.mlp_img(img)
        txt = self.mlp_txt(txt)
        img = gate_mlp.unsqueeze(1) * img
        txt = gate_mlp.unsqueeze(1) * txt
        img = img_res + img
        txt = txt_res + txt
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qk_norm=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = self.mlp_block(hidden_size, mlp_ratio)
        self.ada_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def mlp_block(self, hidden_size: int, mlp_ratio: float) -> nn.Module:
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        return nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_norm(vec).chunk(6, dim=1)

        # attention
        x_res = x
        x = self.norm1(x)
        x = (1 + scale_msa).unsqueeze(1) * x + shift_msa.unsqueeze(1)
        x = self.attn(x, pos_embed=pe)
        x = gate_msa.unsqueeze(1) * x
        x = x_res + x

        # mlp
        x_res = x
        x = self.norm2(x)
        x = (1 + scale_mlp).unsqueeze(1) * x + shift_mlp.unsqueeze(1)
        x = self.mlp(x)
        x = gate_mlp.unsqueeze(1) * x
        x = x_res + x
        return x


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.ada_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        scale, shift = self.ada_norm(vec).chunk(2, dim=1)
        x = self.norm(x)
        x = (1 + scale).unsqueeze(1) * x + shift.unsqueeze(1)
        x = self.linear(x)
        return x
