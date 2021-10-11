import os


def ready_for_torch(assert_=False):
    result = 'CUDA_VISIBLE_DEVICES' in os.environ
    if assert_:
        assert result, 'The environment is not ready for this module yet.'
    return result


def set_cuda_visible_devices(gpus: list):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


def get_cuda_visible_devices():
    return os.environ['CUDA_VISIBLE_DEVICES'].split(',')
