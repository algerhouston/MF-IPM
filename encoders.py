import torch.nn as nn

from crc import ConvReluConv


class REncoder(nn.Module):
    """Shared R-Encoder for target/reference image branches."""

    def __init__(self, in_channels=1, feature_channels=32, depth=2):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth):
            layers.append(ConvReluConv(feature_channels, feature_channels))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
