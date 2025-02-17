import os
import logging

from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate
import numpy as np

import ComoSVC_DiT.utils.audio as audio


def setup_logging(verbose):
    # set logger
    if verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")


def setup_device(args):
    if torch.cuda.is_available() and not args.use_cpu:
        logging.info("Using GPU for inference.")
        device = torch.device("cuda")
    else:
        logging.info("Using CPU for inference.")
        device = torch.device("cpu")
    return device


def compute_mel(wav, audio_config):
    if not isinstance(wav, np.ndarray):
        wav = audio.load_wav(wav, sr=audio_config.sample_rate)
    if audio_config.use_preemphasis:
        wav = audio.preemphasis(wav)
    mel_spectrogram = audio.melspectrogram(wav, audio_config).astype(np.float32).T
    return mel_spectrogram


def load_model(ckpt_path, device, model_name, use_ema=False):
    
    config_file = os.path.dirname(ckpt_path) + "/config.yaml"
    logging.info(f"{model_name} config file: {config_file}.")
    assert os.path.isfile(config_file)
    config = OmegaConf.load(config_file)
    
    model_config = config.task.model

    if 'generator' in model_config:
        if hasattr(model_config, 'global_params') and not hasattr(model_config.generator, 'global_params'):
            model_config.generator.global_params = model_config.global_params
        model_config = model_config.generator
        from ComoSVC_DiT.models.hifigan import HiFiGANGenerator
        model = HiFiGANGenerator(model_config.config)
        ckpt_dict = torch.load(ckpt_path)
        model_ckpt_dict = ckpt_dict['state_dict']['model_gen']
    model.load_state_dict(model_ckpt_dict)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(p.numel() for p in model_parameters)
    logging.info(f"Number of parameters [{model_name}]: {num_params/1.e6}M")
    model.eval().to(device)
    logging.info(f"{model_name} model loaded.")
    try:
        model.remove_weight_norm()
        logging.info("Removed weight_norm")
    except:
        pass

    return model, config

