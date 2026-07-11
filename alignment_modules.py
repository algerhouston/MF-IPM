import torch
import torch.nn as nn


class SpatialAlignmentModule(nn.Module):
    """Spatial alignment module: learns an H x W x 1 spatial gate."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, target_feature):
        spatial_gate = self.gate(target_feature)
        return target_feature * spatial_gate


class ChannelAlignmentModule(nn.Module):
    """Channel alignment module: learns a 1 x 1 x C channel gate."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, reference_feature):
        channel_gate = self.gate(self.gap(reference_feature))
        return reference_feature * channel_gate


class FeatureGateFusion(nn.Module):
    """Element-wise gate fusion used by the alignment branches."""

    def forward(self, target_feature, reference_feature):
        gate = torch.sigmoid(target_feature * reference_feature)
        return target_feature * gate + reference_feature * (1.0 - gate)
