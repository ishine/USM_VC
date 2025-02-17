import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import numpy as np
import commons
import pickle
from scipy.special import softmax
from tqdm import tqdm
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from f0_utils import get_cont_lf0
import torchaudio
import torchaudio.compliance.kaldi as kaldi

def load_pkl(file_path):
    f_read = open(file_path, 'rb')
    dict_pkl = pickle.load(f_read)
    #print(dict2)
    f_read.close()
    return dict_pkl
def dropout1d(myarray, ratio=0.5):
    indices = np.random.choice(np.arange(myarray.size), replace=False,
                               size=int(myarray.size * ratio))
    myarray[indices] = 0
    return myarray
def softmax1(x):
    #max_v = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    #e_x = np.exp(x - max_v)  # subtracts each row with its max value
    e_x = np.exp(x)
    sum_t = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / (sum_t + 1)
    return f_x
def softmax_t(x, t=1.5):
    #t is the temperature
    #max_v = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    #e_x = np.exp(x - max_v)  # subtracts each row with its max value
    e_x = np.exp(x / t)
    sum_t = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / (sum_t)
    return f_x
class TextAudioLoader_ppgmap(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.map_ppg_dict = hparams.map_ppg_dict
        self.map_aug_ppg_dict = hparams.map_aug_ppg_dict
        if is_train:
            self.aug_ratio = hparams.aug_ratio
        else:
            self.aug_ratio = 0.0
        self.ppg_apply_utt_instance_norm = hparams.ppg_apply_utt_instance_norm
        print(f"PPG use Utt-IN: {self.ppg_apply_utt_instance_norm}")

        self.min_text_len = getattr(hparams, "min_text_len", 200)
        self.max_text_len = getattr(hparams, "max_text_len", 2000)

        self.pitch_type = getattr(hparams, "pitch_type", "lf0_uv")

        if self.map_ppg_dict is not None and is_train:
            f_read = open(f'{self.map_ppg_dict}', 'rb')
            self.utt2map_ppg = pickle.load(f_read)
            f_read.close()
        else:
            self.utt2map_ppg = None
        if self.map_aug_ppg_dict is not None and is_train:
            f_read = open(f'{self.map_aug_ppg_dict}', 'rb')
            self.utt2map_aug_ppg = pickle.load(f_read)
            f_read.close()
        else:
            self.utt2map_aug_ppg = None            

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        audiopaths_and_text_new = []
        lengths = []
        print(f"Number of samples before filtering: {len(self.audiopaths_and_text)}")
        for meta_line in self.audiopaths_and_text:
            audio_path, ling_path, pitch_path, \
                f0_path, spec_path, spk_id, style_id, num_frames = meta_line
            num_frames = int(num_frames)
            if num_frames >= self.min_text_len and num_frames <= self.max_text_len:
                lengths.append(num_frames)
                audiopaths_and_text_new.append(meta_line)
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        print(f"Number of samples after filtering: {len(self.audiopaths_and_text)}")

    def align_length(self, ling, spec, wav, pitch, f0):
        min_len = min(
            ling.shape[0],
            spec.shape[1],
            wav.shape[1]//self.hop_length,
            pitch.shape[0],
            f0.shape[0],
        )
        ling = ling[:min_len]
        spec = spec[:, :min_len]
        wav = wav[:, :min_len*self.hop_length]
        pitch = pitch[:min_len]
        f0 = f0[:min_len]
        return ling, spec, wav, pitch, f0

    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        fid_id = os.path.basename(ling_path)[:-4]
        prob = torch.rand(1).item()
        if prob >= self.aug_ratio:
            ling = self.get_ling(ling_path)
        if self.utt2map_ppg is not None and self.utt2map_aug_ppg is not None and prob < self.aug_ratio:
            aug_ppg_types = ['ori','aug']
            aug_type = random.choice(aug_ppg_types)
            if aug_type == 'ori':
                #print(f"using map ppg from ori ppg.")
                ppg_fn = self.utt2map_ppg[fid_id][0]
            elif aug_type == 'aug':
                #print(f"using map ppg from aug ppg.")
                aug_ppgs = self.utt2map_aug_ppg[fid_id]
                random.shuffle(aug_ppgs)
                ppg_fn = aug_ppgs[0]
            ling = self.get_ling(ppg_fn)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )

    def get_f0_pitch(self, f0_path, pitch_path):
        f0 = np.load(f0_path)
        if self.pitch_type == 'lf0_uv':
            uv, lf0 = get_cont_lf0(f0)
            pitch = np.concatenate([lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1).astype(np.float32)
            pitch = torch.from_numpy(pitch).float()
        else:
            # quantized version
            pitch = torch.LongTensor(np.load(pitch_path))
        f0 = torch.from_numpy(f0).float()
        return f0, pitch

    def get_spec(self, spec_path):
        spec = torch.from_numpy(np.load(spec_path)).float()
        return spec

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def utt_mean_std_norm_ppg(self, ppg):
        mean = np.mean(ppg, axis=0, keepdims=True)
        std = np.std(ppg, axis=0, keepdims=True)
        ppg_in = (ppg - mean) / std
        return ppg_in

    def get_ling(self, ling_path):
        #print(f"{ling_path}")
        ling = np.load(ling_path)
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling = self.utt_mean_std_norm_ppg(ling)
        ling = torch.FloatTensor(ling)
        return ling

    def __getitem__(self, index):
        return self.get_feat_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)



class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.aug_ppg_dir = hparams.aug_ppg_dir
        self.aug_ppg_dict = hparams.aug_ppg_dict
        self.hparams = hparams
        if hparams.filter_spk is not None and is_train:
            self.filter_spk = hparams.filter_spk
        else:
            self.filter_spk = None          
        if is_train:
            self.aug_ratio = hparams.aug_ratio
        else:
            self.aug_ratio = 0.0
        self.ppg_apply_utt_instance_norm = hparams.ppg_apply_utt_instance_norm
        print(f"PPG use Utt-IN: {self.ppg_apply_utt_instance_norm}")

        self.min_text_len = getattr(hparams, "min_text_len", 100)#50
        self.max_text_len = getattr(hparams, "max_text_len", 1000)

        self.pitch_type = getattr(hparams, "pitch_type", "lf0_uv")

        if self.aug_ppg_dict is not None:
            f_read = open(f'{self.aug_ppg_dict}', 'rb')
            self.utt2ppg_aug = pickle.load(f_read)
            f_read.close()
        else:
            self.utt2ppg_aug = None

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        audiopaths_and_text_new = []
        lengths = []
        print(f"Number of samples before filtering: {len(self.audiopaths_and_text)}")
        if self.filter_spk is None:
            for meta_line in self.audiopaths_and_text:
                audio_path, ling_path, pitch_path, \
                    f0_path, spec_path, spk_id, style_id, num_frames = meta_line
                num_frames = int(num_frames)
                if num_frames >= self.min_text_len and num_frames <= self.max_text_len:
                    lengths.append(num_frames)
                    audiopaths_and_text_new.append(meta_line)
        else:
            self.filter_spk_set = self.filter_spk.split(',')
            for meta_line in self.audiopaths_and_text:
                audio_path, ling_path, pitch_path, \
                    f0_path, spec_path, spk_id, style_id, num_frames = meta_line
                num_frames = int(num_frames)
                #print(f"spk id: {spk_id} {type(spk_id)}, filter spk: {self.filter_spk} {type(self.filter_spk)}")
                if num_frames >= self.min_text_len and num_frames <= self.max_text_len and spk_id in self.filter_spk_set:
                    lengths.append(num_frames)
                    audiopaths_and_text_new.append(meta_line)                    
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        print(f"Number of samples after filtering: {len(self.audiopaths_and_text)}")

    def align_length(self, ling, spec, wav, pitch, f0):
        min_len = min(
            ling.shape[0],
            spec.shape[1],
            wav.shape[1]//self.hop_length,
            pitch.shape[0],
            f0.shape[0],
        )
        ling = ling[:min_len]
        spec = spec[:, :min_len]
        wav = wav[:, :min_len*self.hop_length]
        pitch = pitch[:min_len]
        f0 = f0[:min_len]
        return ling, spec, wav, pitch, f0

    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        fid_id = os.path.basename(ling_path)[:-4]
        prob = torch.rand(1).item()
        if prob >= self.aug_ratio:
            ling = self.get_ling(ling_path)
        if self.utt2ppg_aug is not None and prob < self.aug_ratio:
            aug_ppgs = self.utt2ppg_aug[fid_id]
            random.shuffle(aug_ppgs)
            ppg_fn = f"{self.aug_ppg_dir}/{aug_ppgs[0]}"
            ling = self.get_ling(ppg_fn)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )

    def get_f0_pitch(self, f0_path, pitch_path):
        f0 = np.load(f0_path)
        if self.pitch_type == 'lf0_uv':
            uv, lf0 = get_cont_lf0(f0)
            pitch = np.concatenate([lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1).astype(np.float32)
            pitch = torch.from_numpy(pitch).float()
        else:
            # quantized version
            pitch = torch.LongTensor(np.load(pitch_path))
        f0 = torch.from_numpy(f0).float()
        return f0, pitch

    def get_spec(self, spec_path):
        spec = torch.from_numpy(np.load(spec_path)).float()
        return spec

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def utt_mean_std_norm_ppg(self, ppg):
        mean = np.mean(ppg, axis=0, keepdims=True)
        std = np.std(ppg, axis=0, keepdims=True)
        ppg_in = (ppg - mean) / std
        return ppg_in

    def get_ling(self, ling_path):
        ling = np.load(ling_path)
        ling = ling.repeat(2, axis=0)
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling = self.utt_mean_std_norm_ppg(ling)
        ling = torch.FloatTensor(ling)
        return ling

    def __getitem__(self, index):
        return self.get_feat_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
class TextAudioLoader_soft(TextAudioLoader):
    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        super().__init__(audiopaths_and_text, hparams, is_train=is_train)
        # if hparams.global_phn_center_path is not None:
        #     self.global_phn_center = np.load(hparams.global_phn_center_path) #dict_size, ppg_dim
        # else:
        #     self.global_phn_center = None
        if hparams.para_softmax_path is not None:
            f =  open(hparams.para_softmax_path, 'rb')
            # weight(w) and bias(b) in the softmax layer in ppg model
            self.para_softmax = pickle.load(f)
            f.close()
        else:
            self.para_softmax = None
        self.soft_map = hparams.soft_map # 
        self.soft_func_type = hparams.soft_func_type
        self.temperature = float(hparams.temperature)
    def get_ling(self, ling_path):
        ling = np.load(ling_path)
        ln1 = ling @ self.para_softmax['w1'].T + self.para_softmax['b1']
        ln1 = np.maximum(0, ln1)
        logit = ln1 @ self.para_softmax['w2'].T + self.para_softmax['b2']
        if self.soft_func_type == 'softmax1':
            logit_soft = softmax1(logit)#T,601
        elif self.soft_func_type == 'softmax_t': 
            logit_soft = softmax_t(logit, t=self.temperature)#T,601
        else:
            logit_soft = softmax(logit, axis=1)#T,400   
        # if self.soft_map:
        #     ling = logit_soft @ self.global_phn_center # T, 256
        # else:
        #     maxi = np.argmax(logit_soft, axis=1) #T,
        #     ling = self.global_phn_center[maxi]
        # #print(f"soft_func_type: {self.soft_func_type}, t: {self.temperature}, soft_map: {self.soft_map}")    
        # if self.ppg_apply_utt_instance_norm:
        #     # print("apply ppg-utt-in")
        #     ling = self.utt_mean_std_norm_ppg(ling)
        logit_soft = logit_soft.repeat(2, axis=0)    
        ling = torch.FloatTensor(logit_soft)
        return ling

class TextAudioLoader_map2phn_Gdict(TextAudioLoader):
    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        super().__init__(audiopaths_and_text, hparams, is_train=is_train)
        if hparams.global_phn_center_path is not None:
            self.global_phn_center = np.load(hparams.global_phn_center_path) #dict_size, ppg_dim
        else:
            self.global_phn_center = None
        if hparams.para_softmax_path is not None:
            f =  open(hparams.para_softmax_path, 'rb')
            # weight(w) and bias(b) in the softmax layer in ppg model
            self.para_softmax = pickle.load(f)
            f.close()
        else:
            self.para_softmax = None
        self.soft_map = hparams.soft_map # 
        self.soft_func_type = hparams.soft_func_type
        self.temperature = float(hparams.temperature)
    def get_ling(self, ling_path):
        ling = np.load(ling_path)
        ln1 = ling @ self.para_softmax['w1'].T + self.para_softmax['b1']
        ln1 = np.maximum(0, ln1)
        logit = ln1 @ self.para_softmax['w2'].T + self.para_softmax['b2']        

        if self.soft_func_type == 'softmax1':
            logit_soft = softmax1(logit)#T,601
        elif self.soft_func_type == 'softmax_t': 
            logit_soft = softmax_t(logit, t=self.temperature)#T,400
        else:
            logit_soft = softmax(logit, axis=1)#T,601    
        if self.soft_map:
            ling = logit_soft @ self.global_phn_center # T, 400
        else:
            maxi = np.argmax(logit_soft, axis=1) #T,
            ling = self.global_phn_center[maxi]
        #print(f"soft_func_type: {self.soft_func_type}, t: {self.temperature}, soft_map: {self.soft_map}")    
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling = self.utt_mean_std_norm_ppg(ling)
        ling = ling.repeat(2, axis=0)    
        ling = torch.FloatTensor(ling)
        return ling
class TextAudioLoader_map2phn_Mixdict(TextAudioLoader_map2phn_Gdict):
    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        super().__init__(audiopaths_and_text, hparams, is_train=is_train)
        self.spk_phn_center_path = hparams.spk_phn_center_path
        self.use_ppg = hparams.use_ppg
        self.shift2Gdict_ratio = hparams.shift2Gdict_ratio  
    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        fid_id = os.path.basename(ling_path)[:-4]
        prob = torch.rand(1).item()
        if prob >= self.aug_ratio:
            ling = self.get_ling(ling_path, spk_id)
        if self.utt2ppg_aug is not None and prob < self.aug_ratio:
            aug_ppgs = self.utt2ppg_aug[fid_id]
            random.shuffle(aug_ppgs)
            ppg_fn = f"{self.aug_ppg_dir}/{aug_ppgs[0]}"
            ling = self.get_ling(ppg_fn, spk_id)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )         
    def get_ling(self, ling_path, spk_id):
        ling = np.load(ling_path)
        ln1 = ling @ self.para_softmax['w1'].T + self.para_softmax['b1']
        ln1 = np.maximum(0, ln1)
        logit = ln1 @ self.para_softmax['w2'].T + self.para_softmax['b2']        
        
        if self.soft_func_type == 'softmax1':
            logit_soft = softmax1(logit)#T,601
        elif self.soft_func_type == 'softmax_t': 
            logit_soft = softmax_t(logit, t=self.temperature)#T,601            
        else:
            logit_soft = softmax(logit, axis=1)#T,601
        spk_phn_center = np.load(f"{self.spk_phn_center_path}/phn_center_{spk_id}.npy") #dict_size, ppg_dim
        if self.soft_map:
            ling_map = logit_soft @ spk_phn_center + self.shift2Gdict_ratio * (logit_soft @ self.global_phn_center) # T, 256
        else:
            maxi = np.argmax(logit_soft, axis=1) #T,
            ling_map = spk_phn_center[maxi] + self.shift2Gdict_ratio * self.global_phn_center[maxi]
        if self.use_ppg:
            #print(f"use_ppg: {self.use_ppg}, soft_func: {self.soft_func_type}")
            ling_map = ling_map + ling    
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling_map = self.utt_mean_std_norm_ppg(ling_map)
        ling_map = ling_map.repeat(2, axis=0)
        ling_map = torch.FloatTensor(ling_map)
        return ling_map          
class TextAudioLoader_map2phn_Mixdict2(TextAudioLoader_map2phn_Gdict):
    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        super().__init__(audiopaths_and_text, hparams, is_train=is_train)
        self.spk_phn_center_path = hparams.spk_phn_center_path
        self.use_ppg = hparams.use_ppg
        self.Gdict_weight = float(hparams.Gdict_weight)
        self.Sdict_weight = float(hparams.Sdict_weight)
        self.ppg_weight = float(hparams.ppg_weight)  
    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        fid_id = os.path.basename(ling_path)[:-4]
        prob = torch.rand(1).item()
        if prob >= self.aug_ratio:
            ling = self.get_ling(ling_path, spk_id)
        if self.utt2ppg_aug is not None and prob < self.aug_ratio:
            aug_ppgs = self.utt2ppg_aug[fid_id]
            random.shuffle(aug_ppgs)
            ppg_fn = f"{self.aug_ppg_dir}/{aug_ppgs[0]}"
            ling = self.get_ling(ppg_fn, spk_id)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )         
    def get_ling(self, ling_path, spk_id):
        ling = np.load(ling_path)
        ln1 = ling @ self.para_softmax['w1'].T + self.para_softmax['b1']
        ln1 = np.maximum(0, ln1)
        logit = ln1 @ self.para_softmax['w2'].T + self.para_softmax['b2']        
        #logit = ling @ self.para_softmax['w'].T + self.para_softmax['b']
        if self.soft_func_type == 'softmax1':
            logit_soft = softmax1(logit)#T,601
        elif self.soft_func_type == 'softmax_t': 
            logit_soft = softmax_t(logit, t=self.temperature)#T,601            
        else:
            logit_soft = softmax(logit, axis=1)#T,601
        spk_phn_center = np.load(f"{self.spk_phn_center_path}/phn_center_{spk_id}.npy") #dict_size, ppg_dim
        if self.soft_map:
            ling_map = self.Sdict_weight * (logit_soft @ spk_phn_center) + self.Gdict_weight * (logit_soft @ self.global_phn_center) # T, 256
        else:
            maxi = np.argmax(logit_soft, axis=1) #T,
            ling_map = self.Sdict_weight * spk_phn_center[maxi] + self.Gdict_weight * self.global_phn_center[maxi]
        if self.use_ppg:
            #print(f"use_ppg: {self.use_ppg}, soft_func: {self.soft_func_type}")
            ling_map = ling_map + self.ppg_weight * ling   
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling_map = self.utt_mean_std_norm_ppg(ling_map)
        ling_map = ling_map.repeat(2, axis=0)
        ling_map = torch.FloatTensor(ling_map)
        return ling_map  
class TextAudioLoader_map2phn_Mixdict3(TextAudioLoader_map2phn_Gdict):
    # cat different phn dict and ppg
    def __init__(self, audiopaths_and_text, hparams, is_train=True):
        super().__init__(audiopaths_and_text, hparams, is_train=is_train)
        self.spk_phn_center_path = hparams.spk_phn_center_path
        #self.use_ppg = hparams.use_ppg
        self.Gdict_weight = float(hparams.Gdict_weight)
        self.Sdict_weight = float(hparams.Sdict_weight)
        self.ppg_weight = float(hparams.ppg_weight) 
        self.mix_type = hparams.mix_type 
    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        fid_id = os.path.basename(ling_path)[:-4]
        prob = torch.rand(1).item()
        if prob >= self.aug_ratio:
            ling = self.get_ling(ling_path, spk_id) #ori ppg
        if self.utt2ppg_aug is not None and prob < self.aug_ratio:
            aug_ppgs = self.utt2ppg_aug[fid_id]
            random.shuffle(aug_ppgs)
            ppg_fn = f"{self.aug_ppg_dir}/{aug_ppgs[0]}"
            ling = self.get_ling(ppg_fn, spk_id) #aug ppg
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )         
    def get_ling(self, ling_path, spk_id):
        ling = np.load(ling_path)
        ln1 = ling @ self.para_softmax['w1'].T + self.para_softmax['b1']
        ln1 = np.maximum(0, ln1)
        logit = ln1 @ self.para_softmax['w2'].T + self.para_softmax['b2']        
        #logit = ling @ self.para_softmax['w'].T + self.para_softmax['b']
        if self.soft_func_type == 'softmax1':
            logit_soft = softmax1(logit)#T,601
        elif self.soft_func_type == 'softmax_t': 
            logit_soft = softmax_t(logit, t=self.temperature)#T,601            
        else:
            logit_soft = softmax(logit, axis=1)#T,601
        if self.spk_phn_center_path is not None:     
            spk_phn_center = np.load(f"{self.spk_phn_center_path}/phn_center_{spk_id}.npy") #dict_size, ppg_dim
        ling_maps = []
        if self.soft_map:
            if self.Sdict_weight > 0:
                ling_maps.append(self.Sdict_weight * (logit_soft @ spk_phn_center)) # T, 256
            if self.Gdict_weight > 0:    
                ling_maps.append(self.Gdict_weight * (logit_soft @ self.global_phn_center))
        else:
            maxi = np.argmax(logit_soft, axis=1) #T,
            if self.Sdict_weight > 0:
                ling_maps.append(self.Sdict_weight * spk_phn_center[maxi]) 
            if self.Gdict_weight > 0:    
                ling_maps.append(self.Gdict_weight * self.global_phn_center[maxi])
        if self.ppg_weight > 0:
            #print(f"use_ppg: {self.use_ppg}, soft_func: {self.soft_func_type}")
            ling_maps.append(self.ppg_weight * ling)
        if self.mix_type == "add":
            ling_map = 0
            for ling in ling_maps:
                ling_map += ling    
        elif self.mix_type == "cat":
            ling_map = np.concatenate(ling_maps, axis=1)
        else:
            raise ValueError("mix_type should be add or act.")    
        if self.ppg_apply_utt_instance_norm:
            # print("apply ppg-utt-in")
            ling_map = self.utt_mean_std_norm_ppg(ling_map)
        ling_map = ling_map.repeat(2, axis=0)
        ling_map = torch.FloatTensor(ling_map)
        return ling_map    
class FbankAudioLoader(TextAudioLoader):
    def __init__(self, audiopaths_and_text, hparams):
        super().__init__(audiopaths_and_text, hparams)
        self.hparams = hparams

    def align_length(self, ling, spec, wav, pitch, f0, fbank):
        min_len = min(
            fbank.shape[0],
            ling.shape[0],
            spec.shape[1],
            wav.shape[1]//self.hop_length,
            pitch.shape[0],
            f0.shape[0],
        )
        ling = ling[:min_len]
        spec = spec[:, :min_len]
        wav = wav[:, :min_len*self.hop_length]
        pitch = pitch[:min_len]
        f0 = f0[:min_len]
        fbank = fbank[:min_len]
        return ling, spec, wav, pitch, f0, fbank

    def fbank(self, audio):
        sr = self.hparams["sampling_rate"]
        waveform = torchaudio.transforms.Resample(orig_freq=sr,
                                                  new_freq=16000)(audio)
        fs = 16000
        mat = kaldi.fbank(audio,
                          num_mel_bins=80,
                          frame_length=25,
                          frame_shift=10,
                          dither=0.0,
                          energy_floor=0.0,
                          sample_frequency=fs)
        speech_lengths = torch.tensor([mat.shape[0]])
        return mat, speech_lengths

    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        ling = self.get_ling(ling_path)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        fbank, fbank_len = self.fbank(wav)
        ling, spec, wav, pitch, f0, fbank = self.align_length(ling, spec, wav, pitch, f0, fbank)
        return (
            ling, spec, wav, pitch, f0, fbank, int(spk_id), int(style_id)
        )
class RepAudioLoader(TextAudioLoader):
    def __init__(self, audiopaths_and_text, hparams):
        super().__init__(audiopaths_and_text, hparams)

    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        ling_path = ling_path.replace("ppg", "rep")
        ling = self.get_ling_rep(ling_path)
        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling, spec, wav, pitch, f0 = self.align_length(ling, spec, wav, pitch, f0)
        return (
            ling, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )

    def get_ling_rep(self, ling_path):
        ling = np.load(ling_path)
        # if self.ppg_apply_utt_instance_norm:
        #     # print("apply ppg-utt-in")
        #     ling = self.utt_mean_std_norm_ppg(ling)
        ling = torch.FloatTensor(ling)
        return ling

    def align_length(self, ling, spec, wav, pitch, f0):
        min_len = min(
            ling.shape[0]* 2,
            spec.shape[1],
            wav.shape[1]//self.hop_length,
            pitch.shape[0],
            f0.shape[0],
        )
        min_len = min_len // 2 * 2
        ling = ling[:min_len // 2]
        spec = spec[:, :min_len]
        wav = wav[:, :min_len*self.hop_length]
        pitch = pitch[:min_len]
        f0 = f0[:min_len]
        return ling, spec, wav, pitch, f0


class RepPPGAudioLoader(TextAudioLoader):
    def __init__(self, audiopaths_and_text, hparams):
        super().__init__(audiopaths_and_text, hparams)

    def get_feat_pair(self, audiopath_and_text):
        audio_path, ling_path, pitch_path, \
            f0_path, spec_path, spk_id, style_id, num_frames = audiopath_and_text
        ling_path1 = ling_path.replace("ppg", "rep")
        ling1 = self.get_ling_rep(ling_path1)
        ling2 = self.get_ling(ling_path)

        wav = self.get_audio(audio_path)
        spec = self.get_spec(spec_path) # [num_bins, num_frames]
        f0, pitch = self.get_f0_pitch(f0_path, pitch_path)
        ling1, ling2, spec, wav, pitch, f0 = self.align_length(ling1, ling2, spec, wav, pitch, f0)
        return (
            ling1, ling2, spec, wav, pitch, f0, int(spk_id), int(style_id)
        )
    def get_ling_rep(self, ling_path):
        ling = np.load(ling_path)
        # if self.ppg_apply_utt_instance_norm:
        #     # print("apply ppg-utt-in")
        #     ling = self.utt_mean_std_norm_ppg(ling)
        ling = torch.FloatTensor(ling)
        return ling

    def align_length(self, ling1, ling2, spec, wav, pitch, f0):
        min_len = min(
            ling1.shape[0]* 2,
            ling2.shape[0],
            spec.shape[1],
            wav.shape[1]//self.hop_length,
            pitch.shape[0],
            f0.shape[0],
        )
        min_len = min_len // 2 * 2
        ling1 = ling1[:min_len // 2]
        ling2 = ling2[:min_len]
        spec = spec[:, :min_len]
        wav = wav[:, :min_len*self.hop_length]
        pitch = pitch[:min_len]
        f0 = f0[:min_len]
        return ling1, ling2, spec, wav, pitch, f0


class FbankAudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, ling_feat_dim=218):
        self.ling_feat_dim = ling_feat_dim

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_ling_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_pitch_len = max([x[3].shape[0] for x in batch])
        max_fbank_len = max([x[5].shape[0] for x in batch])

        if len(batch[0][3].shape) == 2:
            pitch_dim = 2
        else:
            pitch_dim = 1

        ling_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        fbank_lengths = torch.LongTensor(len(batch))

        ling_padded = torch.FloatTensor(len(batch), max_ling_len, self.ling_feat_dim)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        fbank_padded = torch.FloatTensor(len(batch), max_ling_len, 80)

        if pitch_dim == 1:
            pitch_padded = torch.LongTensor(len(batch), max_pitch_len)
        else:
            pitch_padded = torch.FloatTensor(len(batch), max_pitch_len, pitch_dim)

        f0_padded = torch.LongTensor(len(batch), max_pitch_len)

        speaker_ids = torch.LongTensor(len(batch))
        style_ids = torch.LongTensor(len(batch))

        ling_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()
        f0_padded.zero_()
        fbank_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ling = row[0]
            ling_padded[i, :ling.size(0), :] = ling
            ling_lengths[i] = ling.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            pitch = row[3]
            pitch_padded[i, :pitch.size(0)] = pitch

            f0 = row[4]
            f0_padded[i, :f0.size(0)] = f0

            fbank = row[5]
            fbank_padded[i, :fbank.size(0), :] = fbank
            fbank_lengths[i] = fbank.size(0)

            speaker_ids[i] = row[6]
            style_ids[i] = row[7]

        return (
            ling_padded,
            ling_lengths,
            pitch_padded,
            f0_padded,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            fbank_padded,
            fbank_lengths,
            speaker_ids,
            style_ids,
        )


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, ling_feat_dim=218):
        self.ling_feat_dim = ling_feat_dim

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), #x is (ling, spec, wav, pitch, f0, int(spk_id), int(style_id))
            dim=0, descending=True)

        max_ling_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_pitch_len = max([x[3].shape[0] for x in batch])
        
        if len(batch[0][3].shape) == 2:
            pitch_dim = 2
        else:
            pitch_dim = 1

        ling_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        ling_padded = torch.FloatTensor(len(batch), max_ling_len, self.ling_feat_dim)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)#B,1,T
        
        if pitch_dim == 1:
            pitch_padded = torch.LongTensor(len(batch), max_pitch_len)
        else:
            pitch_padded = torch.FloatTensor(len(batch), max_pitch_len, pitch_dim)#B,T,2

        f0_padded = torch.LongTensor(len(batch), max_pitch_len)#B,T

        speaker_ids = torch.LongTensor(len(batch))
        style_ids = torch.LongTensor(len(batch))

        ling_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()
        f0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ling = row[0]
            ling_padded[i, :ling.size(0), :] = ling
            ling_lengths[i] = ling.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            pitch = row[3]
            pitch_padded[i, :pitch.size(0)] = pitch

            f0 = row[4]
            f0_padded[i, :f0.size(0)] = f0

            speaker_ids[i] = row[5]
            style_ids[i] = row[6]

        return (
            ling_padded,
            ling_lengths,
            pitch_padded,
            f0_padded,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            speaker_ids,
            style_ids,
        )


class TextRepAudioCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, ling_feat_dim1, ling_feat_dim2=218):
        self.ling_feat_dim1 = ling_feat_dim1
        self.ling_feat_dim2 =  ling_feat_dim2

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_ling_len1 = max([len(x[0]) for x in batch])
        max_ling_len2 = max([len(x[1]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])
        max_pitch_len = max([x[4].shape[0] for x in batch])

        if len(batch[0][3].shape) == 2:
            pitch_dim = 2
        else:
            pitch_dim = 1

        ling1_lengths = torch.LongTensor(len(batch))
        ling2_lengths = torch.LongTensor(len(batch))

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        ling1_padded = torch.FloatTensor(len(batch), max_ling_len1, self.ling_feat_dim1)
        ling2_padded = torch.FloatTensor(len(batch), max_ling_len2, self.ling_feat_dim2)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        if pitch_dim == 1:
            pitch_padded = torch.LongTensor(len(batch), max_pitch_len)
        else:
            pitch_padded = torch.FloatTensor(len(batch), max_pitch_len, pitch_dim)

        f0_padded = torch.LongTensor(len(batch), max_pitch_len)

        speaker_ids = torch.LongTensor(len(batch))
        style_ids = torch.LongTensor(len(batch))

        ling1_padded.zero_()
        ling2_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        pitch_padded.zero_()
        f0_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            ling1 = row[0]
            ling1_padded[i, :ling1.size(0), :] = ling1
            ling1_lengths[i] = ling1.size(0)

            ling2 = row[1]
            ling2_padded[i, :ling2.size(0), :] = ling2
            ling2_lengths[i] = ling2.size(0)

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            pitch = row[4]
            pitch_padded[i, :pitch.size(0)] = pitch

            f0 = row[5]
            f0_padded[i, :f0.size(0)] = f0

            speaker_ids[i] = row[6]
            style_ids[i] = row[7]

        return (
            ling1_padded,
            ling2_padded,
            ling1_lengths,
            ling2_lengths,
            pitch_padded,
            f0_padded,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            speaker_ids,
            style_ids,
        )

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
