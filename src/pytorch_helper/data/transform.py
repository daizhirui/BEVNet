import torch
import torch.nn.modules as nn

__all__ = ['DeNormalize']


class DeNormalize(nn.Module):
    def __init__(self, mean, std, inplace=False):
        """ The inverse of `torchvision.transforms.Normalize`.

        :param mean: Sequence of means for each channel
        :param std: Sequence of standard deviations for each channel
        :param inplace: Bool to make this operation in-place
        """
        super(DeNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        if not self.inplace:
            tensor = torch.clone(tensor)
        if tensor.ndim == 4:
            for i, m, s in zip([0, 1, 2], self.mean, self.std):
                tensor[:, i].mul_(s).add_(m)
        else:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor
