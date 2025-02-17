import argparse
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import utils
import json
from mel_processing import spectrogram_torch


def get_parser():
    parser = argparse.ArgumentParser(description="Extract pitch information.")
    parser.add_argument(
        '-i',
        '--wav_dir',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--wav_scp',
        type=str,
        default=None,
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
    )
    parser.add_argument(
        "-c",
        '--config',
        type=str,
        default='./configs/train.json',
    )
    parser.add_argument(
        '--nj',
        type=int,
        default=10,
    )
    return parser


def process_one(fid, wav_path, output_dir, hps):

    spec_filename = f"{output_dir}/{fid}.npy"
    if os.path.isfile(spec_filename):
        return 0
    audio, sampling_rate = utils.load_wav_to_torch(wav_path)
    if sampling_rate != hps.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, hps.sampling_rate))
    audio_norm = audio / hps.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm, hps.filter_length,
        hps.sampling_rate, hps.hop_length, hps.win_length,
        center=False)
    spec = torch.squeeze(spec, 0)
    np.save(spec_filename, spec.numpy(), allow_pickle=False)
    return spec.shape[0]


def main(hps, args):

    if args.wav_dir is not None:
        wav_paths = glob.glob(f"{args.wav_dir}/*.wav")
    else:
        assert args.wav_scp is not None
        with open(args.wav_scp, 'r') as f:
            wav_paths = [l.strip().split()[1] for l in f]
    print(f"Found {len(wav_paths)} wav files.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=min(cpu_count(), args.nj))
    futures = []

    for wav_path in tqdm(wav_paths):
        fid = os.path.splitext(os.path.basename(wav_path))[0]
        futures.append(
            executor.submit(
                partial(process_one, fid, wav_path, output_dir, hps)
            )
        )
    results = [f.result() for f in tqdm(futures)]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    config_path = args.config
    with open(config_path, 'r') as f:
        data = f.read()
    config = json.loads(data)
    config = utils.HParams(**config)

    main(config.data, args)

