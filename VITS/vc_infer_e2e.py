import argparse
import os
import glob
import time
import utils
import torch
from tqdm import tqdm
import numpy as np
import json
import logging
from scipy.special import softmax
logging.getLogger('numba').setLevel(logging.WARNING)

from models_nsf import SynthesizerTrn
from models_nsf import SynthesizerTrn_aux
from preprocess_wave import FeatureInput
from f0_utils import get_cont_lf0
#from ensemble_f0_detector import compute_merged_f0s
import pickle


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


def shift_f0(f0, target_f0_stats, how_to_shift):
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", how_to_shift) 
    if how_to_shift == "median":
        src_f0_median = np.median(f0[f0>0])
        trg_f0_median = target_f0_stats['f0_median']
        shift_ratio = trg_f0_median / src_f0_median
        print(f"Source f0 median: {src_f0_median}")
        print(f"Target f0 median: {trg_f0_median}")
        print(f"F0 shift ratio: {shift_ratio}")
        nonzero_idx = np.nonzero(f0)
        f0[nonzero_idx] = f0[nonzero_idx] * shift_ratio
        return f0
    elif "semitone" in how_to_shift:
        if '+' in how_to_shift:
            num_semitones = int(how_to_shift.split('+')[1])
        else:
            num_semitones = -int(how_to_shift.split('-')[1])
        nonzero_indices = np.nonzero(f0)
        f0[nonzero_indices] = f0[nonzero_indices] * 2 ** (num_semitones / 12)
        print(f"Shift # semitones: {num_semitones}.")
        return f0
    elif how_to_shift == "mean":
        src_f0_mean = np.mean(f0[f0>0])
        trg_f0_mean = target_f0_stats['f0_mean']
        shift_ratio = trg_f0_mean / src_f0_mean
        print(f"Source f0 mean: {src_f0_mean}")
        print(f"Target f0 mean: {trg_f0_mean}")
        print(f"F0 shift ratio: {shift_ratio}")
        nonzero_idx = np.nonzero(f0)
        f0[nonzero_idx] = f0[nonzero_idx] * shift_ratio
        return f0
    elif how_to_shift == 'add_minus':
        src_f0_mean = np.mean(f0[f0>0])
        trg_f0_mean = target_f0_stats['f0_mean']
        print(f"Pitch shift: {trg_f0_mean - src_f0_mean}")
        nonzero_idx = np.nonzero(f0)
        f0[nonzero_idx] = f0[nonzero_idx] + trg_f0_mean - src_f0_mean
        return f0
    else:
        print("No pitch shift")
        return f0


def compute_f0_pitch(
    wav_path, 
    sampling_rate,
    target_f0_stats,
    convert=True,
    how_to_shift="median",
):
    f0_src, _, _ = compute_merged_f0s(
        wav_path, sampling_rate, f0_floor=50, f0_ceil=1100, frame_period_ms=10)
    if f0_src is None:
        print("Error occurs when computing F0.")
        return None, None
    f0_src = f0_src.astype(np.float32)
    if not convert:
        uv, cont_lf0 = get_cont_lf0(f0_src)
        lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        return f0_src, lf0_uv
    
    # Shift source f0
    f0_src = shift_f0(f0_src, target_f0_stats, how_to_shift)
    uv, cont_lf0 = get_cont_lf0(f0_src.copy())
    lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
    return f0_src, lf0_uv


def resize2d(x, target_len):
    source = np.array(x)
    source[source<0.001] = np.nan
    target = np.interp(
        np.arange(0, len(source)*target_len, len(source))/ target_len, np.arange(0, len(source)), source)
    res = np.nan_to_num(target)
    return res


def load_lf0_stats(pkl_path, spk_id=0, style_id=0):
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
        for k, v in stats.items():
            if len(v.shape) == 3: 
                stats[k] = v[spk_id][style_id]
            else:
                stats[k] = v[spk_id]
    target_lf0_stats = stats['log_pitch']
    lf0_mean_trg = target_lf0_stats[0]
    lf0_std_trg = target_lf0_stats[1]
    return lf0_mean_trg, lf0_std_trg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "please enter embed parameter ..."
    #parser.add_argument("-w", "--wave", help="input wave", dest="source")
    parser.add_argument("-p", "--ppgs", help="input ppgs", dest="ppgs")
    parser.add_argument("-f", "--f0_dir", help="input f0s", type=str, default=None)
    parser.add_argument("-c", "--ckpt", help="ckpt path", required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="tmp_vc_outputs")
    parser.add_argument("--style_id", type=int, default=0)
    parser.add_argument("--trg_speaker", type=str, default="IDM",)
    parser.add_argument("--how_to_shift", type=str, default=None,)
    parser.add_argument("--speaker_map", type=str, required=True)
    parser.add_argument("--is_ssl", action="store_true")
    parser.add_argument("--is_soft", action="store_true")
    parser.add_argument("--is_aux", action="store_true")
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument("--convert_pitch", action="store_true")
    parser.add_argument("--f0_stats_file", type=str, default=None)
    
    #parser.add_argument("--f0_stats_file", type=str, required=True, help="Pitch statistics file.",)

    args = parser.parse_args()

    if args.how_to_shift == 'null':
        args.how_to_shift = None
    
    exp_dir = os.path.dirname(args.ckpt)
    hps = utils.get_hparams_from_file(f"{exp_dir}/config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.is_aux:
        net_g = SynthesizerTrn_aux(
            ling_dim=hps.data.ling_feat_dim,
            pitch_type=hps.data.pitch_type,
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            **hps.model
        )
    else:
        net_g = SynthesizerTrn(
            ling_dim=hps.data.ling_feat_dim,
            pitch_type=hps.data.pitch_type,
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            **hps.model
        )
    _ = utils.load_checkpoint(args.ckpt, net_g, None)
    _ = net_g.eval().to(device)
    # num_params_p = sum(p.numel() for p in net_g.enc_p.parameters())
    num_params_q = sum(p.numel() for p in net_g.enc_q.parameters())
    # num_params_f = sum(p.numel() for p in net_g.flow.parameters())
    num_params_z = sum(p.numel() for p in net_g.parameters())
    # num_params_dec = sum(p.numel() for p in net_g.dec.parameters())
    #num_params = num_params_p + num_params_flow + num_params_dec
    num_params = num_params_z - num_params_q
    print(f"Number of parameters: {num_params/1.e6}M")

    print(hps.data.sampling_rate)
    print(hps.data.hop_length)
    featureInput = FeatureInput(hps.data.sampling_rate, hps.data.hop_length)
    print(f"Convert pitch: {args.convert_pitch}")

    #load global dict and softmax para
    try:
        global_phn_center = np.load(hps.data.global_phn_center_path) #dict_size, ppg_dim
    except:
        global_phn_center = None
    #load weight(w) and bias(b) in the softmax layer in ppg model
    try:
        with open(hps.data.para_softmax_path, 'rb') as f:
            para_softmax = pickle.load(f)
    except:
        para_softmax = None
        #print(f"no para_softmax found")

    # Load speaker map
    with open(args.speaker_map, 'r') as f:
        speaker2id_map = json.load(f) 
    if hps.data.use_f0: 
        # Load f0 stats
        with open(args.f0_stats_file, 'r') as f:
            f0_stats = json.load(f)        
        target_f0_stats = f0_stats[args.trg_speaker]    

    speaker_id = torch.LongTensor([speaker2id_map[args.trg_speaker]]).to(device)
    style_id = torch.LongTensor([args.style_id]).to(device)
    print(f"Target speaker/singer: {args.trg_speaker}")
    print(f"Target speaker/singer ID: {speaker_id[0].item()}")
    print(f"is_ssl: {args.is_ssl}")
    print(f"is_soft: {args.is_soft}")

    # load spk dict para
    try:
        spk_phn_center_path = hps.data.spk_phn_center_path
    except:
        spk_phn_center_path = None
    Gdict_weight = hps.data.Gdict_weight
    Sdict_weight = hps.data.Sdict_weight
    ppg_weight = hps.data.ppg_weight
    mix_type = hps.data.mix_type
    print(f"Mix_type: {mix_type}, G_w: {Gdict_weight}, S_w: {Sdict_weight}, ppg_w: {ppg_weight}")
    if spk_phn_center_path is not None:
        spk_phn_center = np.load(f"{spk_phn_center_path}/phn_center_{speaker2id_map[args.trg_speaker]}.npy") #dict_size, ppg_dim

    os.makedirs(args.output_dir, exist_ok=True)





    def utt_mean_std_norm_ppg(ppg):
        mean = np.mean(ppg, axis=0, keepdims=True)
        std = np.std(ppg, axis=0, keepdims=True)
        ppg_in = (ppg - mean) / std
        return ppg_in
    if os.path.isdir(f"{args.ppgs}"):
        ppg_flist = glob.glob(f"{args.ppgs}/*.npy")
    elif os.path.isfile(f"{args.ppgs}"):
        ppg_flist = [line.strip().split()[1] for line in open(f"{args.ppgs}")]
    else:
        print(f"ppgs is not dir or file")
    for ppg_file in tqdm(ppg_flist):
        
        try:
            fid = os.path.basename(ppg_file)[:-4]
            save_fname = f"{args.output_dir}/{fid}.wav"
            # if os.path.exists(save_fname):
            #     continue
            if hps.data.use_f0:
                wav_path = f"{args.source}/{fid}.wav"
            ppg_ori = np.load(ppg_file)
            #ppg_ori = ppg_ori.T
            if args.is_ssl and para_softmax is not None:
                ln1 = ppg_ori @ para_softmax['w1'].T + para_softmax['b1']
                ln1 = np.maximum(0, ln1)
                logit = ln1 @ para_softmax['w2'].T + para_softmax['b2']
                logit_soft = softmax(logit, axis=1)#T,601
            elif para_softmax is not None:
                logit = ppg_ori @ para_softmax['w'].T + para_softmax['b']
                logit_soft = softmax(logit, axis=1)#T,601

            #import ipdb; ipdb.set_trace()
            if args.is_soft:
                ppg = logit_soft
            else:
                ppg_maps = []
                if Sdict_weight > 0: 
                    ppg_maps.append(Sdict_weight * (logit_soft @ spk_phn_center)) # T, 256
                if Gdict_weight > 0:
                    ppg_maps.append(Gdict_weight * (logit_soft @ global_phn_center))
                if ppg_weight > 0:
                    ppg_maps.append(ppg_weight * ppg_ori)
                if mix_type == 'add':
                    ppg = 0
                    for ling in ppg_maps:
                        ppg += ling
                elif mix_type == 'cat':
                    ppg = np.concatenate(ppg_maps, axis=1)     
                else:
                    raise ValueError("mix_type should be add or cat.")                
                
            if args.is_ssl:
                ppg = ppg.repeat(2, axis=0)       
            if hps.data.ppg_apply_utt_instance_norm:
                ppg = utt_mean_std_norm_ppg(ppg)
            ppg_len = len(ppg)            
            if hps.data.use_f0:
                if args.how_to_shift is not None:
                    how_to_shift = args.how_to_shift
                else:
                    how_to_shift = 'null'
                f0, lf0_uv = compute_f0_pitch(
                    wav_path,
                    24000,
                    convert=args.convert_pitch,
                    target_f0_stats=target_f0_stats,
                    how_to_shift=how_to_shift
                )
                if f0 is None:
                    print(f"Skip {wav_path} due to F0 extraction error.")
                    continue
                pitch = lf0_uv
                min_len = min(ppg_len, len(pitch))
                ppg = ppg[:min_len]
                pitch = pitch[:min_len]
                f0 = f0[:min_len]
                pitch = torch.from_numpy(pitch).unsqueeze(0).to(device).float()
                f0 = torch.from_numpy(f0).unsqueeze(0).to(device).float()                                
            else:
                f0 = None
                pitch = None    

            ppg_len = torch.LongTensor([ppg_len]).to(device)
            ppg = torch.FloatTensor(ppg).unsqueeze(0).to(device)

            with torch.no_grad():
                audio = (
                    net_g.infer(ppg, ppg_len, pitch, f0, speaker_id=speaker_id, style_id=style_id)[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
            #save_fname = f"{args.output_dir}/{fid}.wav"
            featureInput.save_wav(audio, save_fname)
        except KeyboardInterrupt:
            print("Interrupt!")
            break
