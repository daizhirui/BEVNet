import torch.nn.modules as nn

from .kernels import *


class ConvThreshold(nn.Module):
    kernels = {
        x.__name__: x for x in [
            ScaleAdaptiveOnesKernel, ScaleAdaptiveExpKernel
        ]
    }

    ONES_KERNEL = ScaleAdaptiveOnesKernel.__name__
    EXP_KERNEL = ScaleAdaptiveExpKernel.__name__

    def __init__(self, mask_th, kernel_name, kernel_kwargs):
        super(ConvThreshold, self).__init__()
        self.mask_th = mask_th
        self.kernel_name = kernel_name
        self.kernel_kwargs = kernel_kwargs
        self.non_neg = nn.ReLU(inplace=True).eval()

    def forward(self, bev_map, bev_scale, return_mask=False):
        kernel = self.kernels[self.kernel_name](
            scales=bev_scale, **self.kernel_kwargs
        )
        bev_map_conv = kernel.forward(self.non_neg(bev_map))
        if return_mask:
            mask = bev_map_conv.ge(self.mask_th).float()
            return bev_map_conv, mask
        else:
            return bev_map_conv
