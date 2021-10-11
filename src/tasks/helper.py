import io

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_helper.utils.array import normalize_range
from torchvision.transforms.functional import to_pil_image

matplotlib.use('Agg')


def assemble_3in1_image(image, gt, pred, titles=None):
    # height, width = image.shape[1:]  # CHW
    if titles is None:
        titles = ('input', 'ground truth', 'prediction')
    figure = plt.figure(figsize=(8, 3))
    font_dict = {'fontsize': 5}
    plt.subplot(131)
    plt.imshow(to_pil_image(image.cpu()))
    plt.title(titles[0], fontdict=font_dict)
    plt.axis('off')

    plt.subplot(132)
    gt = gt.clone().detach()
    gt = (normalize_range(gt) * 255).type(torch.uint8)
    plt.imshow(to_pil_image(gt.cpu()), cmap='jet')
    plt.title(titles[1], fontdict=font_dict)
    plt.axis('off')

    plt.subplot(133)
    pred = pred.clone().detach()
    pred = (normalize_range(pred) * 255).type(torch.uint8)
    plt.imshow(to_pil_image(pred.cpu()), cmap='jet')
    plt.title(titles[2], fontdict=font_dict)
    plt.axis('off')

    plt.tight_layout()

    buf = io.BytesIO()
    figure.savefig(buf, format='png', dpi=180)
    plt.close()
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return torch.tensor(np.transpose(img, (2, 0, 1)))
