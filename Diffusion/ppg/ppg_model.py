import torch
import yaml
import numpy as np
import os
import torch.nn.functional as F
import pickle
from scipy.special import softmax
import torchaudio
from .asr_model import init_asr_model
from .wenet.dataset.feats import kaldiFbank
def build_ppg_model(ppg_model_path,ppg_config,device="cpu"):
    with open(ppg_config, 'r') as fin:
        ppg_configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Init asr model from configs
    if not os.path.exists(ppg_configs['cmvn_file']):
        ppg_configs['cmvn_file'] = os.path.join(os.path.dirname(ppg_model_path),"global_cmvn")

    ppg_model = init_asr_model(ppg_configs)
    checkpoint = torch.load(ppg_model_path)
    model_dict = ppg_model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in
            model_dict}
    model_dict.update(pretrained_dict)
    ppg_model.load_state_dict(model_dict)
    ppg_model = ppg_model.to(device).eval()
    return ppg_model

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

class PPGModelWapper(object):
    def __init__(
        self, 
        ppg_model_path,
        ppg_config,
        device,
        output_type="ppg",
        map_mix_ratio=1.0,
        ppg_frame_length=10,
        mel_f_shift = 10,
        global_phn_center_path="",
        para_softmax_path = "",
        ppg_norm=True
    ):
        super().__init__()
        self.ppg_model = build_ppg_model(ppg_model_path,ppg_config,device)
        self.device = device
        self.output_type = output_type
        self.map_mix_ratio = map_mix_ratio
        self.ppg_frame_length = ppg_frame_length
        self.mel_f_shift = mel_f_shift
        self.featCal=  kaldiFbank().eval()
        self.ppg_norm=ppg_norm
        if self.output_type == "map" or self.output_type == "both" or self.output_type == "softmax":
            self.global_phn_center = np.load(global_phn_center_path) #dict_size, ppg_dim
            f =  open(para_softmax_path, 'rb')
            self.para_softmax = pickle.load(f)
            f.close()
            self.global_phn_center = torch.from_numpy(self.global_phn_center).to(device).to(torch.float32)
            self.para_softmax={
                'w': torch.from_numpy(self.para_softmax['w']).to(device),
                'b': torch.from_numpy(self.para_softmax['b']).to(device),
            }
            
            
    def norm_ppg(self, ppg_seq):
        if ppg_seq.dim() == 3:
            assert ppg_seq.shape[0] == 1
            ppg_seq = ppg_seq[0]
            have_batch = True
        else:
            have_batch = False
        device = ppg_seq.device
        ppg_seq = ppg_seq.cpu().numpy()
        mean = np.mean(ppg_seq,0)[None,:]
        std = np.std(ppg_seq,0)[None,:]
        ppg_norm = (ppg_seq - mean) / std
        ppg_norm = torch.from_numpy(ppg_norm).to(device)
        if have_batch:
            ppg_norm = ppg_norm[None,...]
        return ppg_norm 
    
    def ppg_to_target(self, ppg, true_len):
        padding_mask = ~make_pad_mask(true_len)[:,:,None]
        if self.output_type == "map" or self.output_type == "both" :
            logit = ppg @ self.para_softmax['w'].T + self.para_softmax['b']
            logit_soft = logit.softmax(dim=-1)
            map_ppg = logit_soft @ self.global_phn_center 
            if self.map_mix_ratio ==  1.0:
                ret_ppg = map_ppg
            else:
                ret_ppg = ppg *  (1 - self.map_mix_ratio) + map_ppg *self.map_mix_ratio
            ret_ppg = ret_ppg*padding_mask
        elif  self.output_type =="softmax" :
            logit = ppg @ self.para_softmax['w'].T + self.para_softmax['b']
            ret_ppg = logit.softmax(dim=-1)
            ret_ppg = ret_ppg*padding_mask
        elif self.output_type == "ppg":
            ret_ppg = ppg
            ret_ppg = ret_ppg*padding_mask
        if self.ppg_norm:
            ret_ppg = self.norm_ppg(ret_ppg)
        if self.output_type == "both":
            ppg = ppg*padding_mask
            ret_ppg = [ppg,ret_ppg]
        return ret_ppg
    @torch.no_grad()
    def mel_to_ppg(self, mel, mel_lens):
        ppg, logits = self.ppg_model.extract(mel, mel_lens, stream=False)
        true_len =( mel_lens / (self.ppg_frame_length/self.mel_f_shift)).long()
        ppg_len = ppg.shape[1]
        true_len = true_len.clamp(max = ppg_len)
        ppg = self.ppg_to_target(ppg, true_len)
        return ppg, true_len
    
    @torch.no_grad()
    def audio_to_mel(self, audio, sr=None):
        if isinstance(audio, str):
            audio,sr = torchaudio.load(audio)
            audio = audio.to(self.device)
        if len(audio.shape) == 1:
            audio = audio.squeeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
            audio = resampler(audio)
        feats, feats_len = self.featCal(audio)
        feats_len = feats_len.to(self.device)
        return feats, feats_len
    
    @torch.no_grad()
    def audio_to_ppg(self, audio, sr=None):
        feats, feats_len = self.audio_to_mel(audio, sr)
        ppg, true_len = self.mel_to_ppg(feats, feats_len)
        return ppg, true_len