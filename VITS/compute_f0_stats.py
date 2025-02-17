import argparse
import json
import os
import glob
import numpy as np
from collections import defaultdict


parser = argparse.ArgumentParser(description="Compute merged f0 values")
parser.add_argument(
    '-i',
    '--f0_dir',
    type=str,
    required=True,
)
parser.add_argument(
    '--train_list',
    type=str,
    required=True,
)
parser.add_argument(
    '--fid2spk',
    type=str,
    required=True,
)
parser.add_argument(
    '--stat_out',
    type=str,
    required=True,
)
args = parser.parse_args()

train_fids = []
with open(args.train_list, 'r') as f:
    for l in f:
        train_fids.append(
            os.path.basename(l.split('|')[0])[:-4]
        )
train_fids = set(train_fids)

f0_dir = args.f0_dir
target_path = args.stat_out

fid2spk = {}
with open(args.fid2spk, 'r') as f:
    for line in f:
        fid, spk = line.strip().split()
        fid2spk[fid] = spk

file_list = glob.glob(f"{f0_dir}/*.npy")
print(len(file_list))

speaker2filelist = defaultdict(list)
for f in file_list:
    fid = os.path.basename(f)[:-4]
    if fid not in train_fids:
        continue
    if fid not in fid2spk:
        continue
    speaker_name = fid2spk[fid]
    speaker2filelist[speaker_name].append(f)

def compute_f0_stats(f_list):
    f0s = []
    for f in f_list:
        f0 = np.load(f)
        f0s.append(f0[f0>0])
    f0s = np.concatenate(f0s)
    log_f0s = np.log(f0s)
    f0_mean, f0_std = np.mean(f0s), np.std(f0s)
    f0_median = np.median(f0s)
    log_f0_mean, log_f0_std = np.mean(log_f0s), np.std(log_f0s)
    log_f0_median = np.median(log_f0s)
    return {
        "f0_mean": float(f0_mean),
        "f0_std": float(f0_std),
        "f0_median": float(f0_median),
        "log_f0_mean": float(log_f0_mean),
        "log_f0_std": float(log_f0_std),
        "log_f0_median": float(log_f0_median),
    }

f0_stats = {}
for spk, f_list in speaker2filelist.items():
    print(spk, ":", f" {len(f_list)} files.")
    f0_stats[spk] = compute_f0_stats(f_list)

with open(target_path, 'w') as f:
    json.dump(f0_stats, f, indent=4)
