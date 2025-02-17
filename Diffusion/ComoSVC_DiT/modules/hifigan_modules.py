import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from ComoSVC_DiT.modules.base_layers import Conv1d, UpsampleLayer
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        
class ResidualBlock(torch.nn.Module):
    """ 
    Refer to https://github.com/jik876/hifi-gan/blob/master/models.py
    """
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), causal=False):
        super(ResidualBlock, self).__init__()
        self.h = h
        self.relu_slope = 0.1 if self.h is None else self.h.relu_slope
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], causal=causal))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class PostConv(nn.Module):
    def __init__(self, kernel_size=7, filters=128, causal=False):
        super(PostConv, self).__init__()
        self.post_conv1 = weight_norm(
            Conv1d(1, filters, kernel_size, causal=causal))
        self.post_conv2 = weight_norm(
            Conv1d(filters, 1, kernel_size, causal=causal))

    def forward(self, inputs):
        outputs = self.post_conv1(inputs)
        outputs = F.leaky_relu(outputs, 0.1)
        outputs = self.post_conv2(outputs)
        outputs = torch.tanh(outputs)
        return outputs

    def remove_weight_norm(self):
        remove_weight_norm(self.post_conv1)
        remove_weight_norm(self.post_conv2)
