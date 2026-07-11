import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

from meanflow import MeanFlow
from dummy_model import DummyModel  

import torch.nn as nn
from torchvision.transforms import ToTensor



class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, z, t, r, y):
        return self.net(z)  
        
image_path = r""
img = Image.open(image_path).convert('L')  
img = img.resize((256, 256))
tensor_img = ToTensor()(img).unsqueeze(0).to(torch.device('cuda'))  # [1,1,256,256]
tensor_img = tensor_img.clamp(0, 1)

device = torch.device("cuda")
model = SimpleUNet(in_channels=1).to(device)
flow = MeanFlow(channels=1, image_size=256, num_classes=10)

c = torch.randint(0, 10, (1,), device=device)

model.eval()
# flow.eval()

with torch.no_grad():
    loss, mse = flow.loss(model, tensor_img.to(device), c.to(device))
    print("Loss:", loss.item(), "| MSE:", mse.item())

    sampled_imgs = flow.sample_each_class(model, n_per_class=1, sample_steps=1, device=device)

os.makedirs('outputs', exist_ok=True)
save_image(sampled_imgs, '', nrow=5)
save_image(tensor_img, '')
