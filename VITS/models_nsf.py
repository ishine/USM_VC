import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

from hifigan_genertor import HifiGanGenerator
from torchaudio.transforms import MelSpectrogram
from typing import Optional

class TextEncoder(nn.Module):
    def __init__(
        self,
        ling_dim,
        pitch_type,
        out_channels,#192
        hidden_channels,#256
        filter_channels,#768
        n_heads,#2
        n_layers,#6
        kernel_size,#3
        p_dropout,#0.1
        f0=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb_ling = nn.Linear(ling_dim, hidden_channels)
        if f0 == True:
            if pitch_type == 'lf0_uv':
                self.emb_pitch = nn.Linear(2, hidden_channels)
            else:
                self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, ling, pitch, lengths):
        if pitch is None:
            x = self.emb_ling(ling)
        else:    
            x = self.emb_ling(ling) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

class TextEncoder_rep(nn.Module):
    def __init__(
        self,
        ling_dim,
        pitch_type,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        if pitch_type == 'lf0_uv':
            self.emb_pitch = nn.Linear(2, hidden_channels)
        else:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.emb_ling =  ConvTranspose1d(in_channels=ling_dim,
                                         out_channels=hidden_channels,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, ling, pitch, lengths):
        x = self.emb_ling(ling.transpose(1, 2)).transpose(1, 2) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths * 2, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

class TextEncoder_rep_ppg(nn.Module):
    def __init__(
        self,
        ling_dim1,
        ling_dim2,
        pitch_type,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        if pitch_type == 'lf0_uv':
            self.emb_pitch = nn.Linear(2, hidden_channels)
        else:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.emb_ling1 =  ConvTranspose1d(in_channels=ling_dim1,
                                          out_channels=hidden_channels,
                                          kernel_size=2,
                                          stride=2)
        self.emb_ling2 = nn.Linear(ling_dim2, hidden_channels)

    def forward(self, ling1, ling2, pitch, lengths):
        x = self.emb_ling1(ling1.transpose(1, 2)).transpose(1, 2) + self.emb_pitch(pitch)
        x += self.emb_ling2(ling2)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths * 2, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

class TextEncoder_rep_ppg_addnoise(nn.Module):
    def __init__(
        self,
        ling_dim1,
        ling_dim2,
        pitch_type,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        if pitch_type == 'lf0_uv':
            self.emb_pitch = nn.Linear(2, hidden_channels)
        else:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.emb_ling1 =  ConvTranspose1d(in_channels=ling_dim1,
                                          out_channels=hidden_channels,
                                          kernel_size=2,
                                          stride=2)
        self.emb_ling2 = nn.Linear(ling_dim2, hidden_channels)

    def forward(self, ling1, ling2, pitch, lengths):
        ling1 = ling1 + torch.randn_like(ling1)
        x = self.emb_ling1(ling1.transpose(1, 2)).transpose(1, 2) + self.emb_pitch(pitch)
        x += self.emb_ling2(ling2)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths * 2, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        g_spk_channels=0,
        g_style_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    g_spk_channels=g_spk_channels,
                    g_style_channels=g_style_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g_spk=None, g_style=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g_spk=g_spk, g_style=g_style, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g_spk=g_spk, g_style=g_style, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        g_spk_channels=0,
        g_style_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g_spk=None, g_style=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g_spk=g_spk, g_style=g_style)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

class LingPredictor(nn.Module):
    def __init__(
        self,
        ling_dim,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.ling_dim = ling_dim // 3
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.phonemes_predictor = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, 2, kernel_size, p_dropout
        )
        self.linear1 = nn.Linear(hidden_channels, self.ling_dim)

    def forward(self, x, x_mask):
        #print(f"z_size: {x.size()}, hidden_chan: {self.hidden_channels}")
        phonemes_embedding = self.phonemes_predictor(x * x_mask, x_mask)
        # print("x_size:", x.size())
        x1 = self.linear1(phonemes_embedding.transpose(1, 2))
        #x1 = x1.log_softmax(2)
        # print("phonemes_embedding size:", x1.size())
        return x1
    
class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        g_spk_channels=0,
        g_style_channels=0,
        #gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        #if gin_channels != 0:
        #    self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        if g_spk_channels != 0:
            self.spk_cond_layer = nn.Conv1d(g_spk_channels, upsample_initial_channel, 1)
        if g_style_channels != 0:
            self.style_cond_layer = nn.Conv1d(g_style_channels, upsample_initial_channel, 1)

    def forward(self, x, g_spk = None, g_style = None):
        x = self.conv_pre(x)
        if g_spk is not None:
            x = x + self.spk_cond_layer(g_spk)
        if g_style is not None:
            x = x + self.style_cond_layer(g_style)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels

        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,#192
            hidden_channels,#256
            filter_channels,#768
            n_heads,#2
            n_layers,#6
            kernel_size,#3
            p_dropout,#0.1
            f0=False,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": False,
            "audio_sample_rate": 24000,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        #self.dec = HifiGanGenerator(h=hps)
        self.dec = Generator(
            inter_channels,#192
            resblock,#1
            resblock_kernel_sizes,      #[3,7,11]
            resblock_dilation_sizes,    #[[1,3,5], [1,3,5], [1,3,5]]
            upsample_rates,             #[8,5,3,2]
            upsample_initial_channel,   #512
            upsample_kernel_sizes,      #[16,11,7,4]
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
            #gin_channels=gin_channels,
        )        
        # self.dec = Generator(
            # inter_channels,
            # resblock,
            # resblock_kernel_sizes,
            # resblock_dilation_sizes,
            # upsample_rates,
            # upsample_initial_channel,
            # upsample_kernel_sizes,
            # gin_channels=gin_channels,
        # )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )

        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, ling, ling_lengths, pitch, nsff0, y, y_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)

        # z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
        #     z, nsff0, y_lengths, self.segment_size)
        try:
            z_slice, ids_slice = commons.rand_slice_segments(
                z, y_lengths, self.segment_size)
        except:        
            print(f"z:{z.shape}, y_len:{y_lengths}, segment_size:{self.segment_size}")    
           
        #o = self.dec(z_slice, f0=None) #f0_slice
        o = self.dec(z_slice, g_spk=g_spk, g_style=g_style)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ling, ling_lengths, pitch=None, nsff0=None, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        #o = self.dec((z * x_mask)[:, :, :max_len], f0=None) #nsff0
        o = self.dec((z * x_mask)[:, :, :max_len], g_spk=g_spk, g_style=g_style)
        return o, x_mask, (z, z_p, m_p, logs_p)
    
class SynthesizerTrn_Res(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels
        
        self.ling_res = nn.Sequential(
            nn.Linear(ling_dim, 64),
            nn.ReLU(),
            nn.Linear(64, ling_dim),
            nn.ReLU()
        )
            
        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,#192
            hidden_channels,#256
            filter_channels,#768
            n_heads,#2
            n_layers,#6
            kernel_size,#3
            p_dropout,#0.1
            f0=False,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": False,
            "audio_sample_rate": 24000,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        #self.dec = HifiGanGenerator(h=hps)
        self.dec = Generator(
            inter_channels,#192
            resblock,#1
            resblock_kernel_sizes,      #[3,7,11]
            resblock_dilation_sizes,    #[[1,3,5], [1,3,5], [1,3,5]]
            upsample_rates,             #[8,5,3,2]
            upsample_initial_channel,   #512
            upsample_kernel_sizes,      #[16,11,7,4]
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
            #gin_channels=gin_channels,
        )        
        # self.dec = Generator(
            # inter_channels,
            # resblock,
            # resblock_kernel_sizes,
            # resblock_dilation_sizes,
            # upsample_rates,
            # upsample_initial_channel,
            # upsample_kernel_sizes,
            # gin_channels=gin_channels,
        # )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )

        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, ling, ling_lengths, pitch, nsff0, y, y_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]
        
        ling_mask = torch.unsqueeze(commons.sequence_mask(ling_lengths, ling.size(1)), 2).to(
            ling.dtype
        )
        # res ling block
        ling_r = self.ling_res(ling) * ling_mask
        ling = ling + ling_r    

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)

        # z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
        #     z, nsff0, y_lengths, self.segment_size)
        try:
            z_slice, ids_slice = commons.rand_slice_segments(
                z, y_lengths, self.segment_size)
        except:        
            print(f"z:{z.shape}, y_len:{y_lengths}, segment_size:{self.segment_size}")    
           
        #o = self.dec(z_slice, f0=None) #f0_slice
        o = self.dec(z_slice, g_spk=g_spk, g_style=g_style)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ling, ling_lengths, pitch=None, nsff0=None, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]
            
        ling_mask = torch.unsqueeze(commons.sequence_mask(ling_lengths, ling.size(1)), 2).to(
            ling.dtype
        )            
        # res ling block
        ling_r = self.ling_res(ling) * ling_mask
        ling = ling + ling_r
        
        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        #o = self.dec((z * x_mask)[:, :, :max_len], f0=None) #nsff0
        o = self.dec((z * x_mask)[:, :, :max_len], g_spk=g_spk, g_style=g_style)
        return o, x_mask, (z, z_p, m_p, logs_p)       
class SynthesizerTrn_aux(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels

        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": False,
            "audio_sample_rate": 24000,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        #self.dec = HifiGanGenerator(h=hps)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
            #gin_channels=gin_channels,
        )        
        # self.dec = Generator(
            # inter_channels,
            # resblock,
            # resblock_kernel_sizes,
            # resblock_dilation_sizes,
            # upsample_rates,
            # upsample_initial_channel,
            # upsample_kernel_sizes,
            # gin_channels=gin_channels,
        # )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.ling_predictor = LingPredictor(
            ling_dim,
            inter_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )        
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )

        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, ling, ling_lengths, pitch, nsff0, y, y_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        pred_ling = self.ling_predictor(z, y_mask)
        #loss_ppg = F.l1_loss(pred_ppgs * y_mask, ling * y_mask)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)

        # z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
        #     z, nsff0, y_lengths, self.segment_size)
        try:
            z_slice, ids_slice = commons.rand_slice_segments(
                z, y_lengths, self.segment_size)
        except:        
            print(f"z:{z.shape}, y_len:{y_lengths}, segment_size:{self.segment_size}")    
           
        #o = self.dec(z_slice, f0=None) #f0_slice
        o = self.dec(z_slice, g_spk=g_spk, g_style=g_style)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_ling

    def infer(self, ling, ling_lengths, pitch=None, nsff0=None, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        #o = self.dec((z * x_mask)[:, :, :max_len], f0=None) #nsff0
        o = self.dec((z * x_mask)[:, :, :max_len], g_spk=g_spk, g_style=g_style)
        return o, x_mask, (z, z_p, m_p, logs_p)    
class SynthesizerTrn_aux_nospk(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels

        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=False,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": False,
            "audio_sample_rate": 24000,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        #self.dec = HifiGanGenerator(h=hps)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
            #gin_channels=gin_channels,
        )        
        # self.dec = Generator(
            # inter_channels,
            # resblock,
            # resblock_kernel_sizes,
            # resblock_dilation_sizes,
            # upsample_rates,
            # upsample_initial_channel,
            # upsample_kernel_sizes,
            # gin_channels=gin_channels,
        # )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.ling_predictor = LingPredictor(
            ling_dim,
            inter_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )        
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )

        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, ling, ling_lengths, pitch, nsff0, y, y_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=None, g_style=None)
        pred_ling = self.ling_predictor(z, y_mask)
        #loss_ppg = F.l1_loss(pred_ppgs * y_mask, ling * y_mask)
        z_p = self.flow(z, y_mask, g_spk=None, g_style=None)

        # z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
        #     z, nsff0, y_lengths, self.segment_size)
        try:
            z_slice, ids_slice = commons.rand_slice_segments(
                z, y_lengths, self.segment_size)
        except:        
            print(f"z:{z.shape}, y_len:{y_lengths}, segment_size:{self.segment_size}")    
           
        #o = self.dec(z_slice, f0=None) #f0_slice
        o = self.dec(z_slice, g_spk=g_spk, g_style=g_style)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_ling

    def infer(self, ling, ling_lengths, pitch=None, nsff0=None, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        m_p, logs_p, x_mask = self.enc_p(ling, None, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=None, g_style=None, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        #o = self.dec((z * x_mask)[:, :, :max_len], f0=None) #nsff0
        o = self.dec((z * x_mask)[:, :, :max_len], g_spk=g_spk, g_style=g_style)
        return o, x_mask, (z, z_p, m_p, logs_p)     
class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):
        super(SynthesizerTrnMs768NSFsid_nono, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels
        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

        # self.hop_length = hop_length#
        #self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            f0=False,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": True,
            "audio_sample_rate": 24000,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        #self.dec = HifiGanGenerator(h=hps)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
            #gin_channels=gin_channels,
        )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            # The hook we want to remove is an instance of WeightNorm class, so
            # normally we would do `if isinstance(...)` but this class is not accessible
            # because of shadowing, so we check the module name directly.
            # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(self, phone, phone_lengths, pitch, nsff0, y, y_lengths, speaker_id=None, style_id=None):  # 这里ds是id，[bs,1]
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            #g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
            g_spk = self.emb_spk(speaker_id).unsqueeze(-1)
        if self.num_styles > 1:
            assert style_id is not None
            #g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]
            g_style = self.emb_style(style_id).unsqueeze(-1) 
        #g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g_spk=g_spk, g_style=g_style)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        speaker_id: torch.Tensor,
        style_id: torch.Tensor, 
        max_len: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            #g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
            g_spk = self.emb_spk(speaker_id).unsqueeze(-1)
        if self.num_styles > 1:
            assert style_id is not None
            #g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]
            g_style = self.emb_style(style_id).unsqueeze(-1) 
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], g_spk=g_spk, g_style=g_style)
        return o, x_mask, (z, z_p, m_p, logs_p)

class MultiPeriodDiscriminatorV2(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        # periods = [2, 3, 5, 7, 11, 17]
        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SynthesizerTrn_asr(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        sample_rate=24000,
        asr_model=None,

        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size

        self.num_speakers = num_speakers
        self.num_styles = num_styles
        self.g_spk_channels = g_spk_channels
        self.g_style_channels = g_style_channels
        self.spk_embed_dim = g_spk_channels
        self.asr_model = asr_model

        self.enc_p = TextEncoder(
            ling_dim,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        hps = {
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "inter_channels": inter_channels,
            "upsample_rates": upsample_rates,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "upsample_initial_channel": upsample_initial_channel,
            "use_pitch_embed": True,
            "audio_sample_rate": sample_rate,
            "resblock": "1",
            "resblock_dilation_sizes": resblock_dilation_sizes
        }
        self.hps = hps
        self.dec = HifiGanGenerator(h=hps)
        # self.dec = Generator(
            # inter_channels,
            # resblock,
            # resblock_kernel_sizes,
            # resblock_dilation_sizes,
            # upsample_rates,
            # upsample_initial_channel,
            # upsample_kernel_sizes,
            # gin_channels=gin_channels,
        # )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3,
            g_spk_channels=g_spk_channels,
            g_style_channels=g_style_channels,
        )
        self.fbank = MelSpectrogram(sample_rate=16000,
                                    n_mels=80, hop_length=160,
                                    win_length=400
                                    )

        if num_speakers > 1:
            assert g_spk_channels > 0
            self.emb_spk = nn.Embedding(num_speakers, g_spk_channels)
        if num_styles > 1:
            assert g_style_channels > 0
            self.emb_style = nn.Embedding(num_styles, g_style_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, ling, ling_lengths, pitch, nsff0,
                y, y_lengths, fbank, fbank_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        with torch.no_grad():

            # sr = self.hps["audio_sample_rate"]
            # fs = 16000
            # waveform = torchaudio.transforms.Resample(orig_freq=sr,
            #                                           new_freq=fs)(wave)
            # """
            # waveform = waveform * (1 << 15)
            # mat = kaldi.fbank(waveform,
            #                   num_mel_bins=80,
            #                   frame_length=25,
            #                   frame_shift=10,
            #                   dither=0.0,
            #                   energy_floor=0.0,
            #                   sample_frequency=fs)
            # """
            # mat = self.fbank(waveform)
            # device = waveform.device
            # speech_lengths = torch.tensor([mat.shape[0]]).to(device)
            # speech = mat.unsqueeze(0).to(device)

            ppg, logits = self.asr_model.extract(fbank, fbank_lengths, False)
            ling = ppg

        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling, pitch, ling_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)

        z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, nsff0, y_lengths, self.segment_size)
        o = self.dec(z_slice, f0=f0_slice)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ling, ling_lengths, pitch, nsff0, fbank, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        m_p, logs_p, x_mask = self.enc_p(ling, pitch, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        o = self.dec((z * x_mask)[:, :, :max_len], f0=nsff0)
        return o, x_mask, (z, z_p, m_p, logs_p)
class SynthesizerTrn_rep(SynthesizerTrn):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__(ling_dim,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers,
        num_styles,
        g_spk_channels,
        g_style_channels,
        **kwargs)
        del self.enc_p

        self.enc_p = TextEncoder_rep(
            ling_dim,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

class SynthesizerTrn_rep_ppg(SynthesizerTrn):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        ling_dim1,
        ling_dim2,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__(ling_dim1,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers,
        num_styles,
        g_spk_channels,
        g_style_channels,
        **kwargs)
        del self.enc_p

        self.enc_p = TextEncoder_rep_ppg(
            ling_dim1,
            ling_dim2,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

    def forward(self, ling1, ling2,  ling1_lengths, ling2_lengths, pitch,
                nsff0, y, y_lengths, speaker_id=None, style_id=None):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        # linguistic feature encoder
        m_p, logs_p, x_mask = self.enc_p(ling1, ling2, pitch, ling1_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g_spk=g_spk, g_style=g_style)
        z_p = self.flow(z, y_mask, g_spk=g_spk, g_style=g_style)

        z_slice, f0_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z, nsff0, y_lengths, self.segment_size)
        o = self.dec(z_slice, f0=f0_slice)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ling1, ling2, ling_lengths, pitch, nsff0, speaker_id=None, style_id=None, max_len=None,
              noise_scale=0.66):
        g_spk, g_style = None, None
        if self.num_speakers > 1:
            assert speaker_id is not None
            g_spk = F.normalize(self.emb_spk(speaker_id)).unsqueeze(-1) # [b, h, 1]
        if self.num_styles > 1:
            assert style_id is not None
            g_style = F.normalize(self.emb_style(style_id)).unsqueeze(-1) # [b, h, 1]

        m_p, logs_p, x_mask = self.enc_p(ling1, ling2, pitch, ling_lengths)

        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask

        z = self.flow(z_p, x_mask, g_spk=g_spk, g_style=g_style, reverse=True)
        # o = self.dec((z * x_mask)[:, :, :max_len], g=g)
        o = self.dec((z * x_mask)[:, :, :max_len], f0=nsff0)
        return o, x_mask, (z, z_p, m_p, logs_p)

class SynthesizerTrn_rep_ppg_perturb(SynthesizerTrn_rep_ppg):
    def __init__(
        self,
        ling_dim1,
        ling_dim2,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers=1,
        num_styles=1,
        g_spk_channels=0,
        g_style_channels=0,
        **kwargs
    ):

        super().__init__(ling_dim1,
        ling_dim2,
        pitch_type,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        num_speakers,
        num_styles,
        g_spk_channels,
        g_style_channels,
        **kwargs)
        del self.enc_p

        self.enc_p = TextEncoder_rep_ppg_addnoise(
            ling_dim1,
            ling_dim2,
            pitch_type,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
