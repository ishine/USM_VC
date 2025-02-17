from dataclasses import dataclass

@dataclass
class ModelConfig:
    mel_channels: int = 80
    mel_proj_channels: int = 80
    ppg_proj_channels: int = 80
    filter_channels: int = 512
    n_heads: int = 2
    n_dec_layers: int = 2 
    kernel_size: int = 3
    p_dropout: int = 0.1
