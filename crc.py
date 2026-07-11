import torch.nn as nn


class ConvReluConv(nn.Module):
    """CRC block from the figure: Conv + activation + Conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        kernel_size=3,
        activation="relu",
    ):
        super().__init__()
        hidden_channels = hidden_channels or out_channels
        padding = kernel_size // 2

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            act,
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.net(x)
