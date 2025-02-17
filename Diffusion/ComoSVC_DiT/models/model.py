import math
import torch
import torch.nn as nn

from .flow_matching import CFMDecoder

def sequence_mask(length: torch.Tensor, max_length: int = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape

def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/matcha_tts.py
class ComoSVC_DiT(nn.Module):
    def __init__(self, input_channel, mel_channels, mel_proj_channels, ppg_proj_channels,filter_channels, n_heads, n_dec_layers, kernel_size, p_dropout,speak_embedding_dim=256):
        super().__init__()

        self.mel_channels = mel_channels
        hidden_channels = mel_proj_channels+ppg_proj_channels
        gin_channels = hidden_channels
        
        self.spk_proj = nn.Linear(speak_embedding_dim,hidden_channels)
        self.f0_embed = nn.Linear(1, hidden_channels)
        self.unit_embed = nn.Linear(input_channel, ppg_proj_channels)
        self.decoder = CFMDecoder(hidden_channels, mel_channels, filter_channels, n_heads, n_dec_layers, kernel_size, p_dropout, gin_channels,
                                  mel_proj=True, mel_input_channels=mel_channels, mel_proj_channels=mel_proj_channels)


    def forward(self, x, x_lengths, y, y_lengths, f0, spk_embd):
        x_lengths = x_lengths.int()
        y_lengths = y_lengths.int()
        x = self.unit_embed(x)
        f0 = self.f0_embed((1+ f0 / 700).log()) 
        spk_embd = self.spk_proj(spk_embd).unsqueeze(1)
        c = spk_embd+f0
        y_mask = sequence_mask(y_lengths, y.size(1)).unsqueeze(1).to(y.dtype)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(y.transpose(1,2), y_mask, x.transpose(1,2), c)
        
        return diff_loss