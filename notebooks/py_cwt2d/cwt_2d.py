import numpy as np
from functools import lru_cache

from .mother_wavelets import wavelets


class WaveletTransformException(Exception):
    pass


def _get_wavelet_mask(wavelet: str, omega_x: np.array, omega_y: np.array, **kwargs):
    assert omega_x.shape == omega_y.shape

    try:
        return wavelets[wavelet](omega_x, omega_y, **kwargs)

    except KeyError:
        raise WaveletTransformException('Unknown wavelet: {}'.format(wavelet))


@lru_cache(5)
def _create_frequency_plane(image_shape: tuple):
    assert len(image_shape) == 2

    h, w = image_shape
    w_2 = (w - 1) // 2
    h_2 = (h - 1) // 2

    w_pulse = 2 * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
    h_pulse = 2 * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

    xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
    dxx_dyy = abs((xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0]))

    return xx, yy, dxx_dyy


def cwt_2d(x, scales, wavelet, **wavelet_args):
    assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x should be 2D numpy array'

    x_image = np.fft.fft2(x)
    xx, yy, dxx_dyy = _create_frequency_plane(x_image.shape)
    cwt = []
    wav_norm = []

    for scale_val in scales:
        mask = scale_val * _get_wavelet_mask(wavelet, scale_val * xx, scale_val * yy, **wavelet_args)
        cwt.append(np.fft.ifft2(x_image * mask))
        wav_norm.append((np.sum(abs(mask)**2)*dxx_dyy)**(0.5 / (2 * np.pi)))

    cwt = np.stack(cwt, axis=2)
    wav_norm = np.array(wav_norm)

    return cwt, wav_norm
