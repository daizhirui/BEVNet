import glob
import os.path

import cv2
from PIL import Image

__all__ = [
    'video_to_images',
    'images_to_gif',
    'video_to_gif'
]


def video_to_images(video_path, image_dir, prefix='frame', image_ext='png'):
    video_capture = cv2.VideoCapture(video_path)
    still_reading, image = video_capture.read()
    os.makedirs(image_dir, )
    frame_count = 0

    while still_reading:
        cv2.imwrite(
            os.path.join(image_dir, f'{prefix}_{frame_count}.{image_ext}'),
            image
        )

        still_reading, image = video_capture.read()
        frame_count += 1


def images_to_gif(image_dir, image_ext, gif_path, duration, loop=False):
    images = glob.glob(os.path.join(image_dir, f'*.{image_ext}'))
    images.sort()
    frames = [Image.open(image) for image in images]
    frames[0].save(
        gif_path, format='GIF', append_images=frames[1:], save_all=True,
        duration=duration, loop=int(loop)
    )


def video_to_gif(video_path, gif_path, duration=None, num_loop=-1):
    video_capture = cv2.VideoCapture(video_path)
    still_reading, image = video_capture.read()

    frames = []
    while still_reading:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(image))

        still_reading, image = video_capture.read()

    assert len(frames) > 1, 'The video has no more than one frame for GIF.'

    kwargs = dict(
        format='GIF', append_images=frames[1:], save_all=True
    )
    if duration:
        kwargs['duration'] = duration
    if num_loop >= 0:
        kwargs['num_loop'] = num_loop
    frames[0].save(gif_path, **kwargs)
