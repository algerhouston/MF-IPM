import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

from meanflow import MeanFlow
from dummy_model import DummyModel  # 新文件，保存上面模型

# # === 设置环境 ===
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Device:", device)
#
# # === 加载病理图像 ===
# image_path = R"E:\ZHUO\gif-png\p-1.png_ISTA_Net_plus_ratio_10_epoch_50_PSNR_34.65_SSIM_0.9271.bmp"
# img = Image.open(image_path).convert('L')  # 转灰度图像
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),  # [0,1]
# ])
# x = transform(img).unsqueeze(0).to(device)  # shape: (1, 1, 256, 256)
#
# # === 分类标签设置 ===
# c = torch.randint(0, 10, (1,), device=device)
#
# # === 模型和MeanFlow ===
# model = DummyModel(channels=1, image_size=256, num_classes=10).to(device)
# flow = MeanFlow(channels=1, image_size=256, num_classes=10)
#
# # === 损失计算 ===
# loss, mse = flow.loss(model, x, c)
# print(f"Loss: {loss.item():.4f} | MSE: {mse.item():.4f}")
#
# # === 可视化采样 ===
# samples = flow.sample_each_class(model, n_per_class=1, device=device)
# os.makedirs("samples", exist_ok=True)
# save_image(samples, "samples/generated_from_pathology.png", nrow=5)
# print("图像已保存：samples/generated_from_pathology.png")


import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from PIL import Image
import os


# ===================== Step 1: 简单 CNN 模型 =====================
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, z, t, r, y):
        return self.net(z)  # 忽略 t/r/y 简化推理


# ===================== Step 2: MeanFlow 引入 =====================

# ===================== Step 3: 加载病理图像并归一化 =====================
image_path = r"E:\ZHUO\gif-png\p-1.png_ISTA_Net_plus_ratio_10_epoch_50_PSNR_34.65_SSIM_0.9271.bmp"
img = Image.open(image_path).convert('L')  # 灰度图
img = img.resize((256, 256))
tensor_img = ToTensor()(img).unsqueeze(0).to(torch.device('cuda'))  # [1,1,256,256]
tensor_img = tensor_img.clamp(0, 1)

# ===================== Step 4: 初始化模型和 MeanFlow =====================
device = torch.device("cuda")
model = SimpleUNet(in_channels=1).to(device)
flow = MeanFlow(channels=1, image_size=256, num_classes=10)

# 创建 label（0 到 9 之间）
c = torch.randint(0, 10, (1,), device=device)

# ===================== Step 5: 前向推理 =====================
model.eval()
# flow.eval()

with torch.no_grad():
    loss, mse = flow.loss(model, tensor_img.to(device), c.to(device))
    print("Loss:", loss.item(), "| MSE:", mse.item())

    sampled_imgs = flow.sample_each_class(model, n_per_class=1, sample_steps=1, device=device)

# ===================== Step 6: 保存输出结果 =====================
os.makedirs('outputs', exist_ok=True)
save_image(sampled_imgs, 'outputs/meanflow_samples.png', nrow=5)
save_image(tensor_img, 'outputs/input_image.png')
