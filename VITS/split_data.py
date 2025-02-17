import argparse
import os
import random
import json
import argparse
import glob
from tqdm import tqdm
import numpy as np
random.seed(666)


parser = argparse.ArgumentParser(description="default")
parser.add_argument(
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
    '--spec_dir',
    type=str,
    required=True,
)
parser.add_argument(
    '--ppg_dir',
    type=str,
    required=True,
)
parser.add_argument(
    '--f0_dir',
    type=str,
    required=True,
)
parser.add_argument(
    '--pitch_dir',
    type=str,
    default=None,
)
parser.add_argument(
    '--fid2spk',
    type=str,
    required=True,
)
parser.add_argument(
    '--fid2style',
    type=str,
    default=None,
)
parser.add_argument(
    '--speaker_map',
    type=str,
    default=None,
)
parser.add_argument(
    '--style_map',
    type=str,
    default=None,
)
parser.add_argument(
    '--val_size',
    type=int,
    default=32,
    help="number of utterances in the validation set.",
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default="features/filelists",
)

args = parser.parse_args()
if args.pitch_dir is None:
    args.pitch_dir = args.f0_dir

val_size = args.val_size
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
fid2spk = {}
with open(args.fid2spk, 'r') as f:
    for line in f:
        fid, spk = line.strip().split()
        fid2spk[fid] = spk
# with open(args.spk_file, 'r') as f:
    # speakers = [l.strip() for l in f]
print(len(fid2spk.keys()), "sentences")
speakers = list(set(fid2spk.values()))
speakers.sort()
if args.speaker_map is None:
    spk2id_map = dict(zip(speakers, range(len(speakers))))
else:
    spk2id_map = {line.split(': ')[0]: int(line.split(': ')[1]) for line in open(args.speaker_map,'r')}
print(spk2id_map)
with open(f"{args.output_dir}/speaker.map", 'w') as f:
    json.dump(spk2id_map, f, indent=4)

if args.fid2style is not None:    
    fid2style = {}        
    with open(args.fid2style, 'r') as f:
        for line in f:
            fid, style = line.strip().split()
            fid2style[fid] = style  
    styles = list(set(fid2style.values()))
    styles.sort()
    if args.style_map is None:
        style2id_map = dict(zip(styles, range(len(styles))))
    else:
        style2id_map = {line.split(': ')[0]: int(line.split(': ')[1]) for line in open(args.style_map,'r')}
    print(style2id_map)
    with open(f"{args.output_dir}/style.map", 'w') as f:
        json.dump(style2id_map, f, indent=4)        
else:
    dummy_style_id = 0

#wav_list = glob.glob(f"{args.wav_dir}/*.wav")
if args.wav_dir is not None:
    wav_list = glob.glob(f"{args.wav_dir}/**/*.wav")
else:
    assert args.wav_scp is not None
    with open(args.wav_scp, 'r') as f:
        wav_list = [l.strip().split()[1] for l in f]
print(f"Found {len(wav_list)} wav files.")

info_lines = []
for wav_path in tqdm(wav_list):
    fid = os.path.basename(wav_path)[:-4]
    wav_path = os.path.abspath(wav_path)
    ppg_path = os.path.abspath(f"{args.ppg_dir}/{fid}.npy")
    # import ipdb; ipdb.set_trace()
    if not os.path.exists(ppg_path):
        print(f"{ppg_path} not found.")
        continue
    pitch_path = os.path.abspath(f"{args.pitch_dir}/{fid}.npy")
    if not os.path.exists(pitch_path):
        print(f"{pitch_path} not found.")
        continue
    f0_path = os.path.abspath(f"{args.f0_dir}/{fid}.npy")
    if not os.path.exists(f0_path):
        print(f"{f0_path} not found.")
        continue
    spec_path = os.path.abspath(f"{args.spec_dir}/{fid}.npy")
    if not os.path.exists(spec_path):
        print(f"{spec_path} not found.")
        continue
    if fid not in fid2spk.keys():
        continue
    if fid not in fid2style.keys():
        continue    
    spk_id = spk2id_map[fid2spk[fid]]
    if args.fid2style is not None:
        style_id = style2id_map[fid2style[fid]]
    else:
        style_id = dummy_style_id    
    num_frames = np.load(ppg_path).shape[0]
    info_lines.append(
        f"{wav_path}|{ppg_path}|{pitch_path}|{f0_path}|{spec_path}|{spk_id}|{style_id}|{num_frames}"
    )

random.shuffle(info_lines)
with open(f'{args.output_dir}/val.list', 'w') as f:
    for l in info_lines[:val_size]:
        f.write(l + '\n')
with open(f'{args.output_dir}/train.list', 'w') as f:
    for l in info_lines[val_size:]:
        f.write(l + '\n')
