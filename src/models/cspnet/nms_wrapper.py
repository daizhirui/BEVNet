# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import sys

if sys.platform == 'win32':
    from .lib_win32.nms.cpu_nms import cpu_nms
    from .lib_win32.nms.gpu_nms import gpu_nms
else:
    from .lib.nms.cpu_nms import cpu_nms
    from .lib.nms.gpu_nms import gpu_nms


def nms(dets, thresh, usegpu, gpu_id):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if usegpu:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)
