import torch.nn as nn

from crc import ConvReluConv
from mri_ops import data_consistency


class RDecoder(nn.Module):
    """Reconstruction decoder with optional data consistency."""

    def __init__(self, feature_channels=32, out_channels=1, depth=2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                ConvReluConv(feature_channels, feature_channels),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feature, measured_kspace=None, mask=None):
        image = self.net(feature)
        if measured_kspace is not None and mask is not None:
            image = data_consistency(image, measured_kspace, mask)
        return image
