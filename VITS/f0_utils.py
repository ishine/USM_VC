import logging

import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuous_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_cont_lf0(f0, frame_period=10.0, lpf=False, return_cont_f0=False):
    """ Compute continuous log-f0 by linearly interpolation.
    
    Args:
        f0 (np.ndarray): fundamental frequency countour.
        frame_period (float): hop-size/frame-shift in milli-second (ms).
        lpf (bool): whether to conduct low-pass filtering on interpolated f0 countour.
        return_cont_f0 (bool): whether to return continuous f0

    """
    uv, cont_f0 = convert_continuous_f0(f0)
    if lpf:
        cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (frame_period * 0.001)), cutoff=20)
        # deep copy
        cont_lf0_lpf = cont_f0_lpf.copy()
        nonzero_indices = np.nonzero(cont_lf0_lpf)
        cont_lf0_lpf[nonzero_indices] = np.log(cont_f0_lpf[nonzero_indices])
        # cont_lf0_lpf = np.log(cont_f0_lpf)
        if return_cont_f0:
            return uv, cont_lf0_lpf, cont_f0
        return uv, cont_lf0_lpf 
    else:
        nonzero_indices = np.nonzero(cont_f0)
        cont_lf0 = cont_f0.copy()
        cont_lf0[cont_f0>0] = np.log(cont_f0[cont_f0>0])
        if return_cont_f0:
            return uv, cont_lf0, cont_f0
        return uv, cont_lf0


def f0_to_coarse(f0, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= f0_bin - 1 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


def coarse_to_f0(f0_coarse, f0_bin=256, f0_max=900.0, f0_min=50.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    uv = f0_coarse == 1
    f0 = f0_mel_min + (f0_coarse - 1) * (f0_mel_max - f0_mel_min) / (f0_bin - 2)
    f0 = ((f0 / 1127).exp() - 1) * 700
    f0[uv] = 0
    return f0


def compute_mean_std(lf0):
    nonzero_indices = np.nonzero(lf0)
    mean = np.mean(lf0[nonzero_indices])
    std = np.std(lf0[nonzero_indices])
    return mean, std


def f02lf0(f0):
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0
