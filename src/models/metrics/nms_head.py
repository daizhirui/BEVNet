import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import maximum_filter

from models.bevnet.bev_transform import BEVTransform


class NMSHead(nn.Module):
    def __init__(self, nms_size=5, nms_min_val=1e-5):
        super(NMSHead, self).__init__()
        self.nms_size = nms_size
        self.min_val = nms_min_val
        self.transform = BEVTransform()

    def _get_max_coords(self, input_map):
        """ _get_max_coords uses NMS of window size (self.nms_size) and
        threshold (self.min_val) to extract the coordinates of the local
        maximums in the input_map.

        :param input_map: (H, W) tensor of the map
        :return tensor: (2, N) tensor of the local maximum coordinates
        """
        assert input_map.ndim == 2, 'invalid input shape'
        device = input_map.device
        input_map = input_map.cpu().numpy()
        max_map = maximum_filter(input_map, size=self.nms_size, mode='constant')
        max_coords = np.stack(
            ((max_map > self.min_val) & (max_map == input_map)).nonzero()
        )
        return torch.tensor(max_coords).flip(0).to(device)  # flip axes

    def get_max_coords(self, input_maps):
        return [self._get_max_coords(input_map[0]) for input_map in input_maps]

    def forward(self, input_map, bev_scale, bev_center):
        bs = len(input_map)

        # non-maximum suppression
        bev_coords = self.get_max_coords(input_map)
        n_coords = [coord.shape[1] for coord in bev_coords]
        bev_pixels = torch.zeros(bs, 3, max(n_coords)).to(input_map.device)
        bev_pixels[:, 2] = 1
        for i, bev_coord in enumerate(bev_coords):
            bev_pixels[i, :2, :bev_coord.shape[1]] = bev_coord

        # convert to world scale
        bev_size = input_map.shape[2:]
        world_coords = self.transform.bev_coord_to_world_coord(
            bev_size, bev_pixels, bev_scale, bev_center
        )
        world_coords = [
            coord[:2, :n_coords[i]] for i, coord in enumerate(world_coords)
        ]
        return dict(world_coords=world_coords, bev_coords=bev_coords)
