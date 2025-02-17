import os
import glob2
import numpy as np
import io
from tqdm import tqdm
import soundfile
#import resampy
import librosa
import pysptk
import pyreaper
import pyworld

import torch
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def pyworld_dio_f0(
    wav, 
    sampling_rate, 
    f0_floor=80, 
    f0_ceil=600,
    frame_period_ms=5,
):
    wav = wav.astype(np.float64)
    _f0, t = pyworld.dio(
        wav, sampling_rate, f0_floor=f0_floor, f0_ceil=f0_ceil,
        frame_period=frame_period_ms,
    )
    f0 = pyworld.stonemask(wav, _f0, t, sampling_rate)
    return f0.astype(np.float32)


def pyreaper_f0(
    wav,
    sampling_rate,
    f0_floor=80, 
    f0_ceil=600,
    frame_period_ms=5,
):
    """REAPER (Robust Epoch And Pitch EstimatoR)"""
    if wav.dtype != np.int16:
        buffer = io.BytesIO()
        soundfile.write(buffer, wav, sampling_rate, "PCM_16", format="raw")
        wav_int16 = np.frombuffer(buffer.getvalue(), dtype=np.int16)
        buffer.close()
    else:
        wav_int16 = wav
    _,_,_, f0, _ = pyreaper.reaper(
        wav_int16, fs=sampling_rate, 
        minf0=f0_floor, maxf0=f0_ceil, 
        frame_period=frame_period_ms/1000.,
    )
    f0[f0 == -1.0] = 0.
    return f0.astype(np.float32)


def pysptk_rapt_f0(
    wav,
    sampling_rate,
    f0_floor=80, 
    f0_ceil=600,
    frame_period_ms=5,
):
    """RAPT - a robust algorithm for pitch tracking
    Returns:
        f0: array, shape (np.ceil(float(len(x))/hopsize))
    """
    f0 = pysptk.sptk.rapt(
        wav.astype(np.float32)*32768.,  # rapt requires -2**15 ~ 2**15
        fs=sampling_rate,
        hopsize=int(frame_period_ms/1000.0*sampling_rate),
        min=f0_floor,
        max=f0_ceil,
        voice_bias=0.,
    )
    return f0.astype(np.float32)


def pad_f0(f0, trg_len):
    f0_len = len(f0)
    if f0_len == trg_len:
        return f0
    pad_left = (trg_len - f0_len) // 2
    pad_right = trg_len - f0_len - pad_left
    padded_f0 = np.pad(f0, (pad_left, pad_right), 
                       mode='constant')
    return padded_f0


def compute_merged_f0s(
    wavfile_path,
    sampling_rate,
    f0_floor=80, 
    f0_ceil=600,
    frame_period_ms=5,
):
    try:
        wav, sr = soundfile.read(wavfile_path)
        if len(wav) < sr:
            return None, sr, len(wav)
        if sr != sampling_rate:
            # wav = resampy(wav, sr, sampling_rate)
            wav = librosa.resample(wav, sr, sampling_rate)
            sr = sampling_rate
        # Pyworld Dio f0
        dio_f0 = pyworld_dio_f0(wav, sr, f0_floor, f0_ceil, frame_period_ms)
        # Reaper f0
        reaper_f0 = pyreaper_f0(wav, sr, f0_floor, f0_ceil, frame_period_ms)
        # RAPT f0
        rapt_f0 = pysptk_rapt_f0(wav, sr, f0_floor, f0_ceil, frame_period_ms)
        
        max_f0_len = max(
            len(dio_f0), len(reaper_f0), len(rapt_f0),
        )
        dio_f0 = pad_f0(dio_f0, trg_len=max_f0_len)
        reaper_f0 = pad_f0(reaper_f0, trg_len=max_f0_len)
        rapt_f0 = pad_f0(rapt_f0, trg_len=max_f0_len)
        
        # Ensemble by majority vote
        vote_result = (np.stack(
            [dio_f0 > 0, reaper_f0 > 0, rapt_f0 > 0], axis=0
        ).astype(np.float32).mean(axis=0) > 0.5).astype(np.float32)
        f0s_concat = np.stack(
            [dio_f0, reaper_f0, rapt_f0],
            axis=0,
        )
        f0_median = np.median(f0s_concat, axis=0)
        merged_f0 = f0_median * vote_result
        return merged_f0.astype(np.float32), sr, len(wav)
    except:
        print(wavfile_path)
        return None, sr, len(wav)


def to_log_f0(f0):
    nonzero_indices = np.nonzero(f0)
    log_f0 = f0.copy()
    log_f0[nonzero_indices] = np.log(f0[nonzero_indices])
    return log_f0


def process_one(
    wav_file_path,
    args,
    output_dir,
):
    fid = os.path.basename(wav_file_path)[:-4]
    save_fname = f"{output_dir}/{fid}.npy"
    if os.path.isfile(save_fname):
        return
    
    merged_f0, sr, wav_len = compute_merged_f0s(
        wav_file_path, args.sampling_rate,
        args.f0_floor, args.f0_ceil, args.frame_period_ms)
    if merged_f0 is None:
        return
    np.save(save_fname, merged_f0, allow_pickle=False)
    # merged_logf0 = to_log_f0(merged_f0)
    # merged_logf0_audio_rate = torch.nn.functional.interpolate(
        # torch.from_numpy(merged_logf0).view(1, 1, -1), 
        # scale_factor=sr*0.001*args.frame_period_ms,
    # ).squeeze().numpy()
    # assert abs(len(merged_logf0_audio_rate) - wav_len) < 1000
    # np.save(logf0_save_fname, merged_logf0_audio_rate)


def run(args):
    """Compute merged f0 values."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.wav_dir is not None:
        wav_file_list = glob2.glob(f"{args.wav_dir}/**/*.wav")
    else:
        assert args.wav_scp is not None
        with open(args.wav_scp, 'r') as f:
            wav_file_list = [l.strip().split()[1] for l in f]
    print(f"Found {len(wav_file_list)} wav files.")

    # Multi-process worker
    if args.num_workers < 2 :
        for wav_file_path in tqdm(wav_file_list):
            process_one(wav_file_path, args, output_dir)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for wav_file_path in wav_file_list:
                futures.append(executor.submit(
                    partial(
                        process_one, wav_file_path, args, output_dir,
                    )
                ))
            results = [future.result() for future in tqdm(futures)]
    

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Compute merged f0 values")
    parser.add_argument(
        "--wav_dir",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--wav_scp",
        default=None,
        type=str,
    )    
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--frame_period_ms",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--sampling_rate",
        default=24000,
        type=int,
    )
    parser.add_argument(
        "--f0_floor",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--f0_ceil",
        default=1100,
        type=int
    )
    parser.add_argument(
        "--num_workers",
        default=cpu_count(),
        type=int
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    run(args)


if __name__ == "__main__":
    main()   
