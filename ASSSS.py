import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from einops import rearrange
import numpy as np
import os
from functools import partial

# ----------- MeanFlow Support Functions -----------

def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

def stopgrad(x):
    return x.detach()

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)))
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()

# ----------- Dummy Denoising Model -----------

class DummyModel(nn.Module):
    def __init__(self, channels=1, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels + 1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, channels, 3, padding=1)
        )
        self.num_classes = num_classes

    def forward(self, z, t, r, y=None):
        # z: [B,C,H,W], t: [B], r: [B]
        t = rearrange(t, 'b -> b 1 1 1').expand_as(z)
        input = torch.cat([z, t], dim=1)
        return self.encoder(input)

# ----------- MeanFlow Class -----------

class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=256,
        num_classes=10,
        flow_ratio=0.5,
        cfg_ratio=0.2,
        cfg_scale=2.0,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.flow_ratio = flow_ratio
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t_np = np.random.rand(batch_size).astype(np.float32)
        r_np = np.random.rand(batch_size).astype(np.float32)
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        e = torch.randn_like(x)
        x = normalize_to_neg1_1(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        uncond = torch.ones_like(c) * self.num_classes
        if self.w is not None:
            with torch.no_grad():
                u_t = model(z, t, t, uncond)
            v_hat = self.w * v + (1 - self.w) * u_t
        else:
            v_hat = v

        cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
        c = torch.where(cfg_mask, uncond, c)

        model_partial = partial(model, y=c)
        u, dudt = torch.autograd.functional.jvp(
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, device='cuda'):
        model.eval()
        c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels,
                        self.image_size, self.image_size, device=device)

        t = torch.ones((c.shape[0],), device=c.device)
        r = torch.zeros((c.shape[0],), device=c.device)

        z = z - model(z, t, r, c)
        z = unnormalize_to_0_1(z.clip(-1, 1))
        return z

# ----------- Main Training Script -----------

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Toy data
    batch_size = 2
    x = torch.rand(batch_size, 1, 256, 256, device=device)
    c = torch.randint(0, 10, (batch_size,), device=device)

    # Model and Flow
    model = DummyModel(channels=1, num_classes=10).to(device)
    flow = MeanFlow(channels=1, image_size=256, num_classes=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    for step in range(100):
        loss, mse = flow.loss(model, x, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}, MSE: {mse.item():.4f}")

    # Sampling
    # samples = flow.sample_each_class(model, n_per_class=2, device=device)
    #
    # os.makedirs("results", exist_ok=True)
    # save_image(samples, "results/samples.png", nrow=2)
    # print("Saved samples to results/samples.png")

if __name__ == "__main__":
    main()
