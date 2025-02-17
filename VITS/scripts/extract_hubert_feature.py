import os, sys

import glob
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from fairseq import checkpoint_utils
import torchaudio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "hubert_base.pt"
outPath = '/data/vctk/hubert' 
wavPath = 'data/vctk/wav24'

os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, fs = torchaudio.load(wav_path)
    if fs != 16000:
        wav = torchaudio.transforms.Resample(
            orig_freq=fs, new_freq=16000)(wav)
        fs = 16000    
    # wav, fs = sf.read(wav_path)
    assert fs == 16000
    feats = wav.squeeze()
    # feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
print("load model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
print("move model to %s" % device)
if device != "cpu":
    model = model.half()
model.eval()


todo = sorted(list(os.listdir(wavPath)))#[i_part::n_part] #i:0,n:1
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    print("no-feature-todo")
else:
    print("all-feature-%s" % len(todo))
    for idx, wav_path in enumerate(todo):
        #try:
        if wav_path.endswith(".wav"):
            #wav_path = "%s/%s" % (wavPath, file)
            fn = os.path.basename(wav_path)[:-4] + ".npy"
            out_fn = f"{outPath}/{fn}"
            #out_path = "%s/%s" % (outPath, file.replace(".wav", ".npy"))
            #import ipdb; ipdb.set_trace()
            if os.path.exists(out_fn):
                continue

            feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            inputs = {
                "source": feats.half().to(device)
                if device != "cpu"
                else feats.to(device),
                "padding_mask": padding_mask.to(device),
                "output_layer": 7,  # layer 9
            }
            try:
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    #feats = model.final_proj(logits[0])
            except:
                continue
            #import ipdb; ipdb.set_trace()
            #feats = feats.squeeze(0).float().cpu().numpy()
            feats_bn = logits[0].squeeze(0).float().cpu().numpy()
            if np.isnan(feats_bn).sum() == 0:
                np.save(out_fn, feats_bn, allow_pickle=False)
            else:
                print("%s-contains nan" % fn)
            #if idx % n == 0:
            print("now-%s,all-%s,%s,%s" % (len(todo), idx, fn, feats_bn.shape))
        #except:
        #    print(f"erro for {file}")
            #print(traceback.format_exc())
    #print("all-feature-done")
