"""Conditional latent DDPM (epsilon prediction) and differentiable eta=0 DDIM."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def cosine_alpha_bar(steps):
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    target = f / f[0]
    beta = (1 - target[1:] / target[:-1]).clamp(0, 0.999)
    return torch.cat((torch.ones(1, dtype=torch.float64), (1 - beta).cumprod(0))).float()


def time_embedding(t, dim=64):
    frequencies = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    phases = t.float()[:, None] * frequencies[None, :]
    return torch.cat((phases.sin(), phases.cos()), dim=-1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.modulation = nn.Linear(width, width * 2)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition):
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return x + self.mlp((1 + scale) * self.norm(x) + shift)


class ConditionalDenoiser(nn.Module):
    def __init__(self, hidden_dim, width, blocks, condition_dropout):
        super().__init__()
        self.type_embedding = nn.Embedding(2, 16)
        self.input = nn.ModuleList([nn.Linear(hidden_dim, width) for _ in range(2)])
        self.output = nn.ModuleList([nn.Linear(width, hidden_dim) for _ in range(2)])
        self.condition = nn.Sequential(
            nn.Linear(64 + 16 + hidden_dim + 1, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([FiLMResidualBlock(width) for _ in range(blocks)])
        self.condition_dropout = condition_dropout

    def forward(self, zt, t, kind, context, available, drop_condition=False):
        if drop_condition and self.condition_dropout:
            keep = (torch.rand(len(zt), device=zt.device) >= self.condition_dropout).to(zt.dtype)
            context = context * keep[:, None]
            available = available * keep
        types = self.type_embedding.weight[kind].expand(len(zt), -1)
        cond = self.condition(
            torch.cat((time_embedding(t), types, context, available[:, None]), dim=-1)
        )
        x = self.input[kind](zt)
        for block in self.blocks:
            x = block(x, cond)
        return self.output[kind](x)


class LatentDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.denoiser = ConditionalDenoiser(
            args.hidden_dim, args.diffusion_width, args.diffusion_blocks, args.condition_dropout
        )
        self.steps, self.start, self.ddim_steps = (
            args.diffusion_steps,
            args.ddim_start,
            args.ddim_steps,
        )
        self.register_buffer("alpha_bar", cosine_alpha_bar(self.steps))
        self.register_buffer("mean", torch.zeros(2, args.hidden_dim))
        self.register_buffer("scale", torch.ones(2, args.hidden_dim))
        self.register_buffer("stats_ready", torch.tensor(False))
        initial = math.log(args.residual_weight / (1 - args.residual_weight))
        self.residual_logit = nn.Parameter(torch.full((2,), initial))

    @torch.no_grad()
    def fit_stats(self, drug, protein, train_drugs, train_proteins):
        for kind, x, indices in [(0, drug, train_drugs), (1, protein, train_proteins)]:
            subset = x[indices]
            if not len(subset):
                raise ValueError("No training entities for diffusion statistics")
            self.mean[kind].copy_(subset.mean(0))
            self.scale[kind].copy_(subset.std(0, unbiased=False).clamp_min(1e-4))
        self.stats_ready.fill_(True)

    def standardize(self, x, kind):
        if not bool(self.stats_ready):
            raise RuntimeError("Fit latent statistics before using diffusion")
        return (x - self.mean[kind]) / self.scale[kind]

    def noise_loss(self, x, context, available, kind):
        z = self.standardize(x.detach(), kind)
        t = torch.randint(1, self.steps + 1, (len(z),), device=z.device)
        epsilon = torch.randn_like(z)
        alpha = self.alpha_bar[t, None]
        zt = alpha.sqrt() * z + (1 - alpha).sqrt() * epsilon
        # The reconstruction target stays detached; the condition encoder must
        # receive gradients from epsilon prediction as well as downstream DTI.
        prediction = self.denoiser(zt, t, kind, context, available, drop_condition=self.training)
        return F.mse_loss(prediction, epsilon)

    def enhance(self, x, context, available, kind, noise=None):
        z = self.standardize(x, kind)
        if noise is None:
            noise = torch.randn_like(z)
        alpha = self.alpha_bar[self.start]
        zt = alpha.sqrt() * z + (1 - alpha).sqrt() * noise
        times = torch.linspace(self.start, 0, self.ddim_steps + 1).round().long().tolist()
        for t, s in zip(times[:-1], times[1:]):
            time = torch.full((len(x),), t, device=x.device, dtype=torch.long)
            epsilon = self.denoiser(zt, time, kind, context, available)
            at, ass = self.alpha_bar[t], self.alpha_bar[s]
            z0 = (zt - (1 - at).sqrt() * epsilon) / at.sqrt()
            zt = ass.sqrt() * z0 + (1 - ass).sqrt() * epsilon
        restored = self.mean[kind] + self.scale[kind] * zt
        return x + self.residual_logit[kind].sigmoid() * (restored - x)
