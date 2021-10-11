from numbers import Number

import torch
import torch.nn.modules as nn

__all__ = [
    'get_grids',
    'GaussianKernel', 'ScaleAdaptiveOnesKernel', 'ScaleAdaptiveExpKernel'
]


def get_grids(size):
    grids = torch.arange(size)
    grids = torch.meshgrid(grids, grids)
    grids = torch.stack(grids, dim=0)
    return grids


class GaussianKernel(nn.Module):
    def __init__(self, kernel_size, sigma, channels, circular=True):
        super(GaussianKernel, self).__init__()
        grids = get_grids(kernel_size).float()
        mean = (kernel_size - 1) / 2.
        variance = 2 * sigma ** 2
        dist = (grids - mean).pow(2).sum(dim=0)
        param = torch.exp(-dist / variance)
        if circular:
            param[dist > mean * mean] = 0
        param /= param.sum()
        param = param.view(1, 1, kernel_size, kernel_size)
        param = param.expand(-1, channels, -1, -1)
        self.kernel = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False
        ).eval()
        self.kernel.weight.data = param
        self.kernel.weight.requires_grad = False

    def forward(self, x):
        return self.kernel.forward(x)


class ScaleAdaptiveOnesKernel(nn.Module):
    def __init__(self, scales, radius):
        super(ScaleAdaptiveOnesKernel, self).__init__()
        if isinstance(scales, torch.Tensor):
            bs = scales.numel()
        elif isinstance(scales, Number):
            bs = 1
            scales = torch.tensor(scales)
        else:
            bs = len(scales)

        r = (radius / scales).int()
        max_r = r.max().item()
        ks = 2 * max_r + 1
        grids = get_grids(ks).float()
        grids = grids.unsqueeze(0).expand(bs, -1, -1, -1).to(scales.device)
        r2 = (r * r).view(bs, 1, 1)
        center = r.max().item()
        # (bs, ks, ks)
        param = (grids - center).pow(2).sum(dim=1).le(r2).float()
        self.kernel = nn.Conv2d(
            bs, bs, ks, padding=max_r, bias=False, groups=bs
        ).eval()  # groups=bs, depth wise convolution!
        self.kernel.weight.data = param.view(bs, 1, ks, ks)
        self.kernel.weight.requires_grad = False

    def forward(self, x):
        return self.kernel(x.transpose(0, 1)).transpose(0, 1)


class ScaleAdaptiveExpKernel(nn.Module):
    def __init__(self, scales, radius, beta, eta, tau):
        super(ScaleAdaptiveExpKernel, self).__init__()
        bs = scales.numel()
        r = (radius / scales).int()
        max_r = r.max().item()
        ks = 4 * max_r + 1
        grids = get_grids(ks).float()
        grids = grids.unsqueeze(0).expand(bs, -1, -1, -1).to(scales.device)
        # risk = eta * exp(-beta * max(0, d - tau))
        dist = (grids - 2 * max_r).pow(2).sum(dim=1).sqrt(2)
        dist *= scales.view(bs, 1, 1)
        param = (-beta * (dist - tau).clamp(0)).exp() * eta
        self.kernel = nn.Conv2d(
            bs, bs, ks, padding=2 * max_r, bias=False, groups=bs
        ).eval()
        self.kernel.weight.data = param.view(bs, 1, ks, ks)
        self.kernel.weight.requires_grad = False

    def forward(self, x):
        return self.kernel(x.transpose(0, 1)).transpose(0, 1)
