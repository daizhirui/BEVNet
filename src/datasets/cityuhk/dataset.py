import os

import h5py
import torch
from pytorch_helper.data.transform import DeNormalize
from pytorch_helper.utils.debug import get_debug_size
from pytorch_helper.utils.debug import is_debug
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import hflip


class CityUHKBEV(Dataset):
    normalize = Normalize(mean=[0.49124524, 0.47874022, 0.4298056],
                          std=[0.21531576, 0.21034797, 0.20407718])
    denormalize = DeNormalize(mean=[0.49124524, 0.47874022, 0.4298056],
                              std=[0.21531576, 0.21034797, 0.20407718])

    available_keys = [
        'bev_center', 'bev_coord', 'bev_map', 'bev_scale',
        'camera_angle', 'camera_fu', 'camera_fv', 'camera_height',
        'feet_annotation', 'feet_map',
        'head_annotation', 'head_map',
        'image',
        'num_annotations',
        'roi_mask',
        'world_coord',
    ]

    def __init__(self, root, datalist, keys: list, use_augment: bool = True):
        super(CityUHKBEV, self).__init__()

        self.root = root
        self.datalist = datalist

        self.keys = list(set(keys))
        self.load_image = 'image' in self.keys

        self.use_augment = use_augment
        self.do_normalization = True

    @staticmethod
    def to_tensor(x, dtype=None):
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype)

    def augment(self, out):
        if not self.use_augment or torch.rand(1).item() < 0.5:
            return out
        else:
            # random horizontal flip
            for key in ['image', 'feet_map', 'head_map', 'bev_map']:
                if key in out:
                    out[key] = hflip(out[key])
            for key in ['feet_annotation', 'head_annotation']:
                if key in out:
                    n = out['num_annotations'].int()
                    out[key][0, :n] = out['image'].size(-1) - out[key][0, :n]
            return out

    def __getitem__(self, item):
        scene_id = self.datalist[item][0]
        image_id = self.datalist[item][1]

        scene_name = f"scene_{scene_id:03d}"

        scene_file = os.path.join(self.root, f"{scene_name}.h5")
        scene_data = h5py.File(scene_file, "r")

        out = dict(
            image_id=self.to_tensor(image_id),
            scene_id=self.to_tensor(scene_id)
        )

        for key in self.keys:
            out[key] = self.to_tensor(scene_data[key][image_id])

        if self.load_image:
            out['image'] = self.to_tensor(scene_data['image'][image_id]) / 255.
            out = self.augment(out)
            if self.do_normalization:
                out['image'] = self.normalize(out['image'])

        scene_data.close()

        return out

    def __len__(self):
        if is_debug():
            return get_debug_size()
        return len(self.datalist)
