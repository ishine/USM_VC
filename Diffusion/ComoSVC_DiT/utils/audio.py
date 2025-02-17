import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy import signal
import soundfile as sf
from numpy import linalg as LA
import librosa.filters
import parselmouth
from parselmouth.praat import call
import pyworld as pw
import math
import random

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def load_wav(wav_path, sr=24000):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio


def save_wav(wav, path, sample_rate, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, sample_rate)


def down_sample(wav, raw_sr=24000, target_sr=16000):
    return librosa.core.resample(wav, raw_sr, target_sr, res_type='kaiser_best')


def preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1, -preemphasis], [1], x)


def inv_preemphasis(x, preemphasis=0.97):
    return signal.lfilter([1], [1, -preemphasis], x)


def trim_silence(wav, hparams):
    # These params are separate and tunable per dataset.
    unused_trimed, index = librosa.effects.trim(
        wav,
        top_db=hparams.trim_top_db,
        frame_length=hparams.
        trim_fft_size,
        hop_length=hparams.
        trim_hop_size)

    num_sil_samples = int(
        hparams.num_silent_frames * hparams.hop_size)
    # head silence is set as half of num_sil_samples
    start_idx = max(index[0] - int(num_sil_samples//2), 0)
    # tail silence is set as twice of num_sil_samples
    stop_idx = min(index[1] + num_sil_samples*2, len(wav))
    trimmed = wav[start_idx:stop_idx]
    return trimmed


_mel_basis = None
_inv_mel_basis = None


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate,
                               hparams.n_fft,
                               n_mels=hparams.acoustic_dim,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_size,
                        win_length=hparams.win_size)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) /
                                                          (-hparams.min_level_db)) -
                           hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) /
                                                    (-hparams.min_level_db)),
                           0, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_acoustic:
        return ((2 * hparams.max_abs_value) *
                ((S - hparams.min_level_db) / (-hparams.min_level_db)) -
                hparams.max_abs_value)
    else:
        return (hparams.max_abs_value *
                ((S - hparams.min_level_db) / (-hparams.min_level_db)))

def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return (((np.clip(D, -hparams.max_abs_value,
                              hparams.max_abs_value) + hparams.max_abs_value)
                     * -hparams.min_level_db / (2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db /
                     hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_acoustic:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db /
                 (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    # Convert back to linear
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)

    return _griffin_lim(S ** hparams.power, hparams)

def _griffin_lim(S, hparams):
    '''
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_size,
                        win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y,
                         hop_length=hparams.hop_size,
                         win_length=hparams.win_size)

def melspectrogram(wav, hparams, compute_energy=False):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        S = _normalize(S, hparams)
    
    if compute_energy:
        energy = np.linalg.norm(np.abs(D), axis=0)
        return S.astype(np.float32), energy.astype(np.float32)
    else:
        return S

def extract_energy(wav, hparams):
    D = _stft(wav, hparams)
    magnitudes = np.abs(D).T
    return LA.norm(magnitudes, axis=1)

def differenceFunction_scipy(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]

    Faster implementation of the difference function.
    The required calculation can be easily evaluated by Autocorrelation function or similarly by convolution.
    Wiener–Khinchin theorem allows computing the autocorrelation with two Fast Fourier transforms (FFT), with time complexity O(n log n).
    This function use an accellerated convolution function fftconvolve from Scipy package.

    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """
    x = np.array(x, np.float64)
    w = x.size
    x_cumsum = np.concatenate((np.array([0]), (x * x).cumsum()))
    conv = fftconvolve(x, x[::-1])
    tmp = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
    return tmp[:tau_max]


def differenceFunction(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]

    Fastest implementation. Use the same approach than differenceFunction_scipy.
    This solution is implemented directly with Numpy fft.


    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).

    This corresponds to equation (8) in [1]

    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """

    cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float)  # scipy method
    return np.insert(cmndf, 0, 1)


def compute_yin(sig, sr, w_len=2048, w_step=256):
    """

    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.

    :param sig: Audio signal (list of float)
    :param sr: sampling rate (int)
    :param w_len: size of the analysis window (samples)
    :param w_step: size of the lag between two consecutives windows (samples)
    :returns:
    """
    dp = []
    tau_max = w_len - 1

    timeScale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    frames = [sig[t:t + w_len] for t in timeScale]

    for i, frame in enumerate(frames):
        df = differenceFunction(frame, w_len, tau_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        dp.append(cmdf)
    return np.array(dp)


def c_m(m, sr):
    """
    midi-to-lag conversion function
    :param m: midi
    :param sr: sample rate
    :return: lag
    """
    return sr / (440 * (2 ** ((m - 69) / 12)))


def yt_m(m, sr, dp, t):
    """
    Yingram time-lag to midi-scale
    """
    c = c_m(m, sr)
    c_c = math.ceil(c)
    c_f = math.floor(c)
    return (dp[t][c_c] - dp[t][c_f]) / (c_c - c_f) * (c - c_f) + dp[t][c_f]


def yingram(wav, hparams):
    """
     Based on the YIN alorgorithm [1]: De Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. The Journal of the Acoustical Society of America, 111(4), 1917-1930.
    """
    dp = compute_yin(wav, hparams.sample_rate, w_len=hparams.win_size)
    ym = []
    for t in range(dp.shape[0]):
        ym1 = []
        for d in range(19, 85):
            """
            midi & tau & Hz (sr=24000)
            18   1037    23
            19   979    24  √
            20   924    25
            ...
            84  22  1046    √
            """
            ytm = yt_m(d, hparams.sample_rate, dp, t)
            ym1.append(ytm)
        ym.append(ym1)
    return np.array(ym)


def formant_shifting(wav, hparams):
    """
    https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    :param sound: numpy.ndarray[numpy.float64]
    :param hparams:
    :return: numpy.ndarray
    """
    sound = parselmouth.Sound(values=wav, sampling_frequency=hparams.sample_rate)
    if hparams.prfs == "old":
        formant_factor = random.uniform(1, hparams.formant_factor)
        take_reciprocal = random.uniform(0, 1) > 0.5
        formant_factor = 1 / formant_factor if take_reciprocal else formant_factor
        return call(sound, "Change speaker", 75, 600, formant_factor, 1, 1, 1).values.T.squeeze(1)
    else:
        formant_shifting_ratio = random.uniform(1, hparams.formant_factor)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            formant_shifting_ratio = 1 / formant_shifting_ratio

        sound_new = apply_formant_and_pitch_shift(
            sound,
            formant_shift_ratio=formant_shifting_ratio,
        )
        return sound_new.values.T.squeeze(1)

def pitch_monotoize(wav, hparams):
    """
    https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    :param sound: numpy.ndarray[numpy.float64]
    :param hparams:
    :return: numpy.ndarray
    """
    sound = parselmouth.Sound(values=wav, sampling_frequency=hparams.sample_rate)
    if hparams.prfs == "old":
        picth_shift_factor = random.uniform(1, hparams.picth_shift_factor)
        take_reciprocal1 = random.uniform(0, 1) > 0.5
        picth_shift_factor = 1 / picth_shift_factor if take_reciprocal1 else picth_shift_factor
        picth_range_factor = random.uniform(1, hparams.picth_range_factor)
        take_reciprocal2 = random.uniform(0, 1) > 0.5
        picth_range_factor = 1 / picth_range_factor if take_reciprocal2 else picth_range_factor
        return call(sound, "Change speaker", 75, 600, 1, 1, 0.0, 1).values.T.squeeze(1)
    else:
        pitch_shift_ratio = random.uniform(1, hparams.picth_shift_factor)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            pitch_shift_ratio = 1 / pitch_shift_ratio

        pitch_range_ratio = random.uniform(1, hparams.picth_range_factor)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            pitch_range_ratio = 1 / pitch_range_ratio

        sound_new = apply_formant_and_pitch_shift(
            sound,
            pitch_shift_ratio=pitch_shift_ratio,
            pitch_range_ratio=pitch_range_ratio,
            duration_factor=1.
        )
        return sound_new.values.T.squeeze(1)


def pitch_randomization(wav, hparams):
    """
    https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    :param sound: numpy.ndarray[numpy.float64]
    :param hparams:
    :return: numpy.ndarray
    """
    sound = parselmouth.Sound(values=wav, sampling_frequency=hparams.sample_rate)
    if hparams.prfs == "old":
        picth_shift_factor = random.uniform(1, hparams.picth_shift_factor)
        take_reciprocal1 = random.uniform(0, 1) > 0.5
        picth_shift_factor = 1 / picth_shift_factor if take_reciprocal1 else picth_shift_factor
        picth_range_factor = random.uniform(1, hparams.picth_range_factor)
        take_reciprocal2 = random.uniform(0, 1) > 0.5
        picth_range_factor = 1 / picth_range_factor if take_reciprocal2 else picth_range_factor
        return call(sound, "Change speaker", 75, 600, 1, picth_shift_factor, picth_range_factor, 1).values.T.squeeze(1)
    else:
        pitch_shift_ratio = random.uniform(1, hparams.picth_shift_factor)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            pitch_shift_ratio = 1 / pitch_shift_ratio

        pitch_range_ratio = random.uniform(1, hparams.picth_range_factor)
        use_reciprocal = random.uniform(-1, 1) > 0
        if use_reciprocal:
            pitch_range_ratio = 1 / pitch_range_ratio

        sound_new = apply_formant_and_pitch_shift(
            sound,
            pitch_shift_ratio=pitch_shift_ratio,
            pitch_range_ratio=pitch_range_ratio,
            duration_factor=1.
        )
        return sound_new.values.T.squeeze(1)


def PEQ(wav, hparams):
    """
    random frequency shaping.
    """
    if hparams.prfs == "old":
        def _random_Q():
            """
            random quality factor
            """
            qmin = 2.0
            qmax = 5.0
            z = random.uniform(0, 1)
            return qmin * ((qmax - qmin) ** z)

        def _random_G():
            """
            random gain
            """
            return random.uniform(-12, 12)

        beg = np.log10(60)
        end = np.log10(10000)
        s = (end - beg) / 9
        f = []
        for i in range(8):
            f.append(10 ** (beg + (i + 1) * s))
        wav = _low_shelving_filer(wav, 60, _random_G(), _random_Q(), sampling_rate=hparams.sample_rate)
        for ff in f:
            wav = _peaking_filer(wav, ff, _random_G(), _random_Q(), sampling_rate=hparams.sample_rate)
        processed = _high_shelving_filer(wav, 10000, _random_G(), _random_Q(), sampling_rate=hparams.sample_rate)
        return processed
    else:
        return parametric_equalizer(wav,hparams.sample_rate)


def _peaking_filer(xin, fpeak=1000, gain=0., Q=1.0, sampling_rate=16000):
    gain = 10 ** (gain / 20)

    def set_peaking(fpeak, gain0, Q0, sr):
        omega = (fpeak / sr) * np.pi * 2.0
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2.0 * Q0)
        A = np.sqrt(gain0)
        b = np.zeros(3)
        a = np.zeros(3)
        if gain0 == 1.0:
            a[0] = 1.0
            b[0] = 1.0
            return b, a
        a[0] = 1.0 + alpha / A
        a[1] = -2.0 * cs
        a[2] = 1.0 - alpha / A
        b[0] = 1.0 + alpha * A
        b[1] = -2.0 * cs
        b[2] = 1.0 - alpha * A
        b /= a[0]
        a /= a[0]

        return b, a

    b, a = set_peaking(fpeak, gain, Q, sr=sampling_rate)
    return signal.lfilter(b, a, xin)


def _high_shelving_filer(xin, fc=1000, gain=0., slope=1.0, sampling_rate=16000):
    gain = 10 ** (gain / 20)

    def set_highshelving(fc, Q, gain, fs):
        A = np.sqrt(gain)
        wc = 2 * np.pi * fc / fs
        wS = np.sin(wc)
        wC = np.cos(wc)
        beta = np.sqrt(A) / Q
        a0 = ((A + 1.0) - ((A - 1.0) * wC) + (beta * wS))
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] = A * ((A + 1.0) + ((A - 1.0) * wC) + (beta * wS)) / a0
        b[1] = -2.0 * A * ((A - 1.0) + ((A + 1.0) * wC)) / a0
        b[2] = A * ((A + 1.0) + ((A - 1.0) * wC) - (beta * wS)) / a0
        a[0] = 1
        a[1] = 2.0 * ((A - 1.0) - ((A + 1.0) * wC)) / a0
        a[2] = ((A + 1.0) - ((A - 1.0) * wC) - (beta * wS)) / a0
        return b, a

    b, a = set_highshelving(fc, gain=gain, Q=slope, fs=sampling_rate)
    return signal.lfilter(b, a, xin)


def _low_shelving_filer(xin, fc=1000, gain=0., slope=1.0, sampling_rate=16000):
    gain = 10 ** (gain / 20)

    def set_lowshelving(fc, Q, gain, fs):
        A = np.sqrt(gain)
        wc = 2 * np.pi * fc / fs
        wS = np.sin(wc)
        wC = np.cos(wc)
        beta = np.sqrt(A) / Q
        a0 = ((A + 1.0) + ((A - 1.0) * wC) + (beta * wS))
        b = np.zeros(3)
        a = np.zeros(3)
        b[0] = A * ((A + 1.0) - ((A - 1.0) * wC) + (beta * wS)) / a0
        b[1] = 2.0 * A * ((A - 1.0) - ((A + 1.0) * wC)) / a0
        b[2] = A * ((A + 1.0) - ((A - 1.0) * wC) - (beta * wS)) / a0
        a[0] = 1
        a[1] = -2.0 * ((A - 1.0) + ((A + 1.0) * wC)) / a0
        a[2] = ((A + 1.0) + ((A - 1.0) * wC) - (beta * wS)) / a0
        return b, a

    b, a = set_lowshelving(fc, gain=gain, Q=slope, fs=sampling_rate)
    return signal.lfilter(b, a, xin)


def extract_pitch(input_wav, hparams):
    def interpolate_f0(data):

        data = np.reshape(data, (data.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return ip_data, vuv_vector

    pitch, _ = pw.dio(input_wav.astype(np.float64), hparams.sample_rate,
                      frame_period=hparams.hop_size / hparams.sample_rate * 1000)
    pitch = pitch.astype(np.float32)
    pitch = interpolate_f0(pitch)[0].reshape([-1])
    pitch = np.array(pitch).reshape([-1])
    mel_pitch = 2595. * np.log10(1. + pitch / 700.)
    return mel_pitch


def norm_f0(input_pitch, mean, std):
    mel_pitch = np.where(input_pitch != 0, (input_pitch - mean) / (std + 1e-6), 0.)
    return mel_pitch


def denorm_f0(input_pitch, mean, std):
    mel_pitch = np.where(input_pitch != 0, input_pitch * (std + 1e-6) + mean, 0.)
    return mel_pitch


def utt_f0_norm(input_wav, hparams):
    mel_pitch = extract_pitch(input_wav, hparams)
    mean = np.mean(mel_pitch)
    std = np.std(mel_pitch)
    mel_pitch = norm_f0(mel_pitch, mean, std)
    return np.expand_dims(mel_pitch, axis=1)


def apply_formant_and_pitch_shift(
        sound: parselmouth.Sound,
        formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
        pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
        pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
        duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT) -> parselmouth.Sound:
    r"""uses praat 'Change Gender' backend to manipulate pitch and formant
        'Change Gender' function: praat -> Sound Object -> Convert -> Change Gender
        see Help of Praat for more details

        # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """

    # pitch = sound.to_pitch()
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.:
        try:
            pitch = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
            # pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
            try:
                pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
                if not math.isnan(pitch_median):
                    new_pitch_median = pitch_median * pitch_shift_ratio
                    if math.isnan(new_pitch_median):
                        new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                else:
                    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
            except:
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
        except:
            new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    try:
        new_sound = parselmouth.praat.call(
            sound, "Change gender", 75, 600,
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            duration_factor
        )
    except Exception as e:
        try:
            new_sound = parselmouth.praat.call(
                sound, "Change gender", 75, 600,
                formant_shift_ratio,
                0.0,
                pitch_range_ratio,
                duration_factor
            )
        except:
            new_sound = sound
    return new_sound



# implemented using the cookbook https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
def lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
    a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def apply_iir_filter(wav, ftype, dBgain, cutoff_freq, sample_rate, Q):
    if ftype == 'low':
        b0, b1, b2, a0, a1, a2 = lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'high':
        b0, b1, b2, a0, a1, a2 = highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'peak':
        b0, b1, b2, a0, a1, a2 = peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    else:
        raise NotImplementedError
    wav_numpy = wav
    b = np.asarray([b0, b1, b2])
    a = np.asarray([a0, a1, a2])
    zi = scipy.signal.lfilter_zi(b, a) * wav_numpy[0]
    return_wav, _ = scipy.signal.lfilter(b, a, wav_numpy, zi=zi)
    return return_wav

def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)
# peq
def parametric_equalizer(wav, sr):
    cutoff_low_freq = 60.
    cutoff_high_freq = 10000.

    q_min = 2
    q_max = 5

    num_filters = 8 + 2  # 8 for peak, 2 for high/low
    key_freqs = [
        power_ratio(float(z) / num_filters, cutoff_low_freq, cutoff_high_freq)
        for z in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]
    Qs = [
        power_ratio(random.uniform(0, 1), q_min, q_max)
        for _ in range(num_filters)
    ]

    # peak filters
    for i in range(1, 9):
        wav = apply_iir_filter(
            wav,
            ftype='peak',
            dBgain=gains[i],
            cutoff_freq=key_freqs[i],
            sample_rate=sr,
            Q=Qs[i]
        )

    # high-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='high',
        dBgain=gains[-1],
        cutoff_freq=key_freqs[-1],
        sample_rate=sr,
        Q=Qs[-1]
    )

    # low-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='low',
        dBgain=gains[0],
        cutoff_freq=key_freqs[0],
        sample_rate=sr,
        Q=Qs[0]
    )

    return wav
