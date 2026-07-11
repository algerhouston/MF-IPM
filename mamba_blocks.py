import torch
import torch.nn as nn
import torch.nn.functional as F

from crc import ConvReluConv
from alignment_modules import ChannelAlignmentModule, SpatialAlignmentModule


class StateSpaceBlock(nn.Module):

    def __init__(self, channels, expansion=2, kernel_size=7):
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden,
        )
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        signal, gate = self.in_proj(x).chunk(2, dim=1)
        signal = self.depthwise(signal)
        x = F.silu(signal) * torch.sigmoid(gate)
        return residual + self.out_proj(x)


class CrossModalFusionMamba(nn.Module):

    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.spatial_alignment = SpatialAlignmentModule(channels)
        self.channel_alignment = ChannelAlignmentModule(channels)
        self.pre_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[StateSpaceBlock(channels) for _ in range(num_blocks)]
        )
        self.post = nn.Sequential(
            ConvReluConv(channels, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, spatial_feature, frequency_feature):
        spatial_feature = self.spatial_alignment(spatial_feature)
        frequency_feature = self.channel_alignment(frequency_feature)
        product_gate = torch.sigmoid(spatial_feature * frequency_feature)
        target_stream = spatial_feature * product_gate
        reference_stream = frequency_feature * (1.0 - product_gate)
        fused = self.pre_fuse(torch.cat([target_stream, reference_stream], dim=1))
        fused = self.blocks(fused)
        return self.post(fused)


class FrequencySpatialHomogenizationMamba(nn.Module):

    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.spatial_ssm = nn.Sequential(
            *[StateSpaceBlock(channels) for _ in range(num_blocks)]
        )
        self.frequency_ssm = nn.Sequential(
            *[StateSpaceBlock(channels) for _ in range(num_blocks)]
        )
        self.mix = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.out_ssm = StateSpaceBlock(channels)

    def forward(self, spatial_feature, frequency_feature):
        spatial = self.spatial_ssm(spatial_feature)
        frequency = self.frequency_ssm(frequency_feature)
        cross_gate = torch.sigmoid(spatial * frequency)
        spatial_h = spatial * cross_gate
        frequency_h = frequency * (1.0 - cross_gate)
        fused = self.mix(torch.cat([spatial_h, frequency_h], dim=1))
        return self.out_ssm(fused)
