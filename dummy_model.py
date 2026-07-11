import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, channels=1, image_size=256, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, 3, padding=1),
        )
        self.num_classes = num_classes

    def forward(self, z, t, r, y):
        return self.encoder(z)
