import os
import argparse
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
import random
from dataclasses import asdict
import random
from dataclasses import asdict
from ComoSVC_DiT.models.ComoSVC import ComoSVC
from ComoSVC_DiT.utils.utils import SpkEmbeddingFinder
from ComoSVC_DiT.utils.vocoder_utils import load_model
import  ComoSVC_DiT.utils.mel_utils  as ut
from ComoSVC_DiT.models.model_config import ModelConfig
from ppg.ppg_model import PPGModelWapper

mel_hps = ut.get_hparams()
audio_config = mel_hps.audio
def wav_to_mel(wav_path, device="cpu"):
    # mel
    mel = ut.compute_mel(wav_path, audio_config)
    mel = torch.from_numpy(mel).to(device)
    return mel[None,:]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def infer(model, processer_models,output_path, device,cfg,source_audio_path,target_spk_name, out_sample_rate=24000,sample_steps=30):
    vocoder, ppg_model,spk_embedding_finder= processer_models
    ppg ,ppg_len= ppg_model.audio_to_ppg(source_audio_path)
    ppg = ppg.to(device)
    source_mel = wav_to_mel(
        wav_path=source_audio_path,
        device=device
    )
    min_len =min(ppg.shape[1],source_mel.shape[1])
    ppg = ppg[:,:min_len]
    source_mel = source_mel[:,:min_len]
    dvec = spk_embedding_finder.find_spk_embedding(target_spk_name,device = device)#[1,256]
    predict_mel = model(
        ppg,
        None,
        source_mel,
        torch.tensor([source_mel.shape[1]]).to(source_mel.device),
        dvec,
        infer = True,
        sample_steps=sample_steps
    ).transpose(-1,-2)
    output_wav = vocoder(predict_mel)
    if isinstance(output_wav, tuple):
        output_wav, subband_outputs = output_wav
    output_wav = output_wav[0]
    output_wav_dir = os.path.dirname(output_path)
    if not os.path.exists(output_wav_dir):
        os.makedirs(output_wav_dir,exist_ok=True)
    torchaudio.save(output_path, output_wav.cpu(), 24000)
    
def build_model(cfg,checkpoint,device):
    model_config = ModelConfig(
        mel_proj_channels = cfg.model.mel_proj_channels,
        ppg_proj_channels = cfg.model.ppg_proj_channels,
        filter_channels = cfg.model.filter_channels,
        n_heads = cfg.model.n_heads,
        n_dec_layers = cfg.model.n_dec_layers,
    )
    model = ComoSVC(
        input_channel=cfg.get(cfg.input_type).ppg_dim,
        speak_embedding_dim = cfg.speak_embedding_dim,
        dit_config=asdict(model_config)
    )
    state_dict = torch.load(checkpoint, map_location='cpu')
    missing_keys,unexpected_keys=model.load_state_dict(state_dict['model'], strict=False)
    print("missing_keys",missing_keys)
    print("unexpected_keys",unexpected_keys)
    model = model.to(device)
    model.eval()
    return model

def build_data_pocesser(cfg,args,device,ppg_norm=False):
    vocoder_model, vocoder_config = load_model(
        args.vocoder_ckpt,
        device,
        'vocoder'
    )

    ppg_model = PPGModelWapper(
        ppg_model_path = cfg.ppg_model.ppg_model_path,
        ppg_config = cfg.ppg_model.ppg_config,
        output_type = cfg.input_type,
        map_mix_ratio = cfg.map.mix_ratio,
        global_phn_center_path = cfg.ppg_model.global_phn_center_path,
        para_softmax_path = cfg.ppg_model.para_softmax_path,
        ppg_norm=ppg_norm,
        device = device
    )
    
    spk_embedding_finder = SpkEmbeddingFinder(
        spk_dict_path = cfg.spk_dict_path
    )
    processer_models = [vocoder_model,ppg_model,spk_embedding_finder]
    return processer_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/ComoSVC_DiT/libritts_usm/model.checkpoint")
    parser.add_argument("--vocoder_ckpt", type=str, default="pretrained_models/vocoder/model.ckpt")
    parser.add_argument("--source_audio", type=str, default="example/example.wav")
    parser.add_argument("--output_path", type=str, default="example/output.wav")
    parser.add_argument('--sample_steps', type=int, default=30,help='sample steps')
    parser.add_argument("--target_spk_name", type=str, default="libritts_4948")
    parser.add_argument("--infer_config", type=str, default="configs/ppg_config.yaml")
    args=parser.parse_args()
    device= f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join(os.path.dirname(args.ckpt),"config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(cfg,OmegaConf.load(args.infer_config))
    set_seed(42)
    sample_steps = args.sample_steps
    
    model = build_model(cfg,args.ckpt,device)
    ppg_norm =cfg.input_norm
    processer_models = build_data_pocesser(cfg,args,device,ppg_norm=ppg_norm)
    infer(
        model = model,
        processer_models = processer_models,
        output_path = args.output_path,
        device = device,
        cfg = cfg,
        source_audio_path = args.source_audio,
        target_spk_name = args.target_spk_name,
        sample_steps=sample_steps
    )
    
if __name__=="__main__":
    main()