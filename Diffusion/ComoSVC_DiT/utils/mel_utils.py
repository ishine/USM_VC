import os
import numpy as np
import ComoSVC_DiT.utils.audio as audio
import argparse
import json
import yaml


def get_hparams(init=True):
    config_path = "configs/audio_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
    hparams = HParams(**config)
    return hparams

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def compute_mel(wav, audio_config):
    if not isinstance(wav, np.ndarray):
        wav = audio.load_wav(wav, sr=audio_config.sample_rate)
    if audio_config.use_preemphasis:
        wav = audio.preemphasis(wav)
    mel_spectrogram = audio.melspectrogram(wav, audio_config).astype(np.float32).T
    return mel_spectrogram




