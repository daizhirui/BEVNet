import os

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from cv2 import cv2
from pytorch_helper.utils.array import normalize_range
from pytorch_helper.utils.array import to_numpy
from pytorch_helper.utils.image import overlay_images
from pytorch_helper.utils.image import to_heatmap
from pytorch_helper.utils.io import imsave


def merge_head_map_and_feet_map(head_map, feet_map):
    map_size = None
    if head_map is not None:
        head_map = to_numpy(head_map)
        # head_map = np.clip(head_map, 0, np.max(head_map) * 0.8)
        head_map = normalize_range(head_map)
        map_size = list(head_map.shape)
    if feet_map is not None:
        feet_map = to_numpy(feet_map)
        # feet_map = np.clip(feet_map, 0, np.max(feet_map) * 0.8)
        feet_map = normalize_range(feet_map)
        map_size = list(feet_map.shape)
    assert map_size is not None
    heatmap = np.zeros(map_size + [3])
    weight_map = np.zeros(map_size)
    if feet_map is not None:
        feet_heatmap = to_heatmap(feet_map, cmap=cv2.COLORMAP_RAINBOW)
        feet_heatmap[feet_map < 1e-5, 0] = 0
        heatmap += feet_heatmap
        weight_map += feet_map
    if head_map is not None:
        head_heatmap = to_heatmap(head_map, cmap=cv2.COLORMAP_JET)
        head_heatmap[head_map < 1e-5, 2] = 0
        heatmap += head_heatmap
        weight_map += head_map

    heatmap = np.clip(heatmap, 0, 1)
    # for i in range(3):
    #     heatmap[..., i] = normalize_range(heatmap[..., i])
    return heatmap, weight_map


def overlay_heatmap_on_image(image, heatmap, weight_map, im_weight=0.95):
    weight_map = np.clip(weight_map, 0, np.mean(weight_map[weight_map > 0]))
    weight_map = normalize_range(weight_map)
    weight_map = weight_map[..., np.newaxis] * im_weight / 2 + (1 - im_weight)
    return overlay_images(
        [image, heatmap],
        [1 - weight_map, weight_map]
    )


def save_density_map(
    path1, density_map, path2=None, image=None, iv_roi=None, texts=None
):
    heatmap = to_heatmap(density_map)
    if path1:
        # save density map
        imsave(path1, (heatmap * 255).astype(np.uint8))
    if path2:
        # save density map + image
        assert image is not None, "image is required to synthesize"
        image = normalize_range(image)
        syn_image = overlay_heatmap_on_image(image, heatmap, density_map)
        if iv_roi is not None:
            syn_image[iv_roi, 0] = 1.0
        syn_image = Image.fromarray((syn_image * 255).astype(np.uint8))
        if texts is not None:
            for x, y, s in texts:
                d1 = ImageDraw.Draw(syn_image)
                font_dir = os.path.dirname(__file__)
                font_path = os.path.join(font_dir, 'SourceCodePro-Bold.ttf')
                my_font = ImageFont.truetype(font_path, 20)
                d1.text((x, y), s, font=my_font, fill=(0, 0, 0), stroke_width=3,
                        stroke_fill=(255, 255, 255))
        imsave(path2, syn_image)


def save_maskmap(path1, maskmap, path2=None, image=None, iv_roi=None):
    if path1:
        imsave(path1, (maskmap * 255).astype(np.uint8))
    if path2:
        assert image is not None, "image is required to synthesize"
        weight_map = (maskmap + 1.) / 2
        syn_image = weight_map[..., np.newaxis] * image
        if iv_roi is not None:
            syn_image[iv_roi, 0] = 1.0  # red channel to 1.0
        imsave(path2, (syn_image * 255).astype(np.uint8))
