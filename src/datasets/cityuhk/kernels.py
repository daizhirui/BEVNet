import torch
import torch.nn.modules as nn

__all__ = [
    'get_grids',
    'GaussianKernel'
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
