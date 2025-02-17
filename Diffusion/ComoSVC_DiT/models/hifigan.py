from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from omegaconf import DictConfig

from ComoSVC_DiT.modules.pqmf import PQMF
from ComoSVC_DiT.modules.torch_stft import STFT
from ComoSVC_DiT.modules.base_layers import Conv1d, UpsampleLayer
from ComoSVC_DiT.modules.hifigan_modules import ResidualBlock, PostConv


class HiFiGANGenerator(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_blocks = len(config.upsample_factors)
        self.num_residuals = len(config.resblock_kernel_sizes)

        self.init_conv = weight_norm(
            Conv1d(
                config.c_dim,
                config.initial_channel,
                config.delay_kernel_size,
                causal=not config.delay,
            )
        )
        # Upsample Nets
        self.upsample_nets = nn.ModuleList()
        for i in range(self.num_blocks):
            in_channel = config.upsample_channels[i-1] if i != 0 else config.initial_channel
            out_channel = config.upsample_channels[i]
            self.upsample_nets.append(
                UpsampleLayer(
                    in_channel,
                    out_channel,
                    config.upsample_kernel_sizes[i],
                    stride=config.upsample_factors[i],
                    activation=None,
                    causal=config.causal,
                    repeat=config.upsample_repeat,
                )
            )
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            res_channel = config.upsample_channels[i]
            for j in range(self.num_residuals):
                self.residual_blocks.append(
                    ResidualBlock(
                        config,
                        res_channel,
                        config.resblock_kernel_sizes[j],
                        config.resblock_dilation_sizes,
                        causal=config.causal,
                    )
                )
        output_channel = config.num_bands if not config.use_istft else (self.get_istft_params()[0] * 2 + 1)*2
        self.final_conv = weight_norm(
            Conv1d(
                res_channel,
                output_channel,
                config.default_kernel_size,
                causal=config.causal,
            )
        )
        # For iSTFTNet
        if config.use_istft:
            hop_length, win_length, filter_length = self.get_istft_params()
            self.istft_padding = torch.nn.ReflectionPad1d((1, 0))
            self.stft = STFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length)
            self.istft_hop_length = hop_length

        # For subband model 
        if config.num_bands > 1:
            self.pqmf = PQMF(config.num_bands)
            self.post_conv = PostConv(config.post_kernel, config.post_filter, causal=config.causal)

    def get_istft_params(self):
        hop_length = self.config.hop_size // reduce(lambda x, y: x * y, self.config.upsample_factors)
        win_length = hop_length * 4
        filter_length = hop_length * 4
        assert reduce(lambda x, y: x * y, self.config.upsample_factors) * hop_length == self.config.hop_size
        return hop_length, win_length, filter_length

    def integrity_test(self):
        hp = self.hps
        if self.config.use_istft: 
            assert self.config.num_bands == 1 # TODO, not support now
        else:
            assert len(self.config.upsample_factors) == \
                len(self.config.upsample_kernel_sizes) == len(self.config.upsample_channels)
            assert reduce(lambda x, y: x * y, self.config.upsample_factors) * self.config.num_bands == self.config.hop_size

    def forward(self, x: torch.Tensor, **kwargs):
        # x shape: [bz,mel_dim, frame] wav: [bz,1,tlen]
        x = self.init_conv(x)
        for i in range(self.num_blocks):
            x = F.leaky_relu(x, self.config.relu_slope)
            x = self.upsample_nets[i](x)
            for j in range(self.num_residuals):
                if j == 0:
                    xs = self.residual_blocks[i * self.num_residuals + j](x)
                else:
                    xs += self.residual_blocks[i * self.num_residuals + j](x)
            x = xs / self.num_residuals
        x = F.leaky_relu(x)
        x = self.final_conv(x)

        if self.config.use_istft:
            x = self.istft_padding(x)
            spec = torch.exp(x[:,:self.istft_hop_length * 2 + 1, :])
            phase = torch.sin(x[:, self.istft_hop_length * 2 + 1:, :])
            x = self.stft.inverse(spec, phase)
        else:
            x = torch.tanh(x) #(B, subbands, T // subbands)

        if self.config.num_bands > 1:
            x = self.pqmf.synthesis(x) # (B, 1, T)
            x = self.post_conv(x) # (B, 1, T)
        return x

    def remove_weight_norm(self):
        for l in self.upsample_nets:
            l.remove_weight_norm()
        for l in self.residual_blocks:
            l.remove_weight_norm()
        remove_weight_norm(self.init_conv)
        remove_weight_norm(self.final_conv)
        if self.config.num_bands > 1:
            self.post_conv.remove_weight_norm()
