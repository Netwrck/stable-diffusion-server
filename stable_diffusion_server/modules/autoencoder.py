from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import ResBlock, SamePadConv2d, AttentionBlock


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


class Encoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.num_resolutions = len(params.ch_mult)
        self.num_res_blocks = params.num_res_blocks
        self.resolution = params.resolution
        self.in_channels = params.in_channels

        # downsampling
        self.conv_in = SamePadConv2d(params.in_channels, params.ch, kernel_size=3)

        curr_res = params.resolution
        in_ch_mult = (1,) + tuple(params.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = params.ch * in_ch_mult[i_level]
            block_out = params.ch * params.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SamePadConv2d(block_out, block_out, 3, stride=2)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_out, out_channels=block_out)
        self.mid.attn_1 = AttentionBlock(block_out)
        self.mid.block_2 = ResBlock(in_channels=block_out, out_channels=block_out)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.conv_out = SamePadConv2d(block_out, 2 * params.z_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.num_resolutions = len(params.ch_mult)
        self.num_res_blocks = params.num_res_blocks
        self.resolution = params.resolution
        self.in_channels = params.in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = params.ch * params.ch_mult[self.num_resolutions - 1]
        curr_res = params.resolution // 2 ** (self.num_resolutions - 1)
        self.z_channels = params.z_channels

        # z to block_in
        self.conv_in = SamePadConv2d(params.z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttentionBlock(block_in)
        self.mid.block_2 = ResBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = params.ch * params.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, kernel_size=3, stride=2, padding=1, output_padding=1)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = SamePadConv2d(block_in, params.out_ch, kernel_size=3)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        x = 2 * x - 1
        z = self.encoder(x)
        z = self.scale_factor * z
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        z = self.decoder(z)
        z = (z + 1) / 2
        return z
