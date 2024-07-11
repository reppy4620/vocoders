import torch
import torch.nn as nn
from vocoders.layers.convnext import ConvNeXtLayer


class WaveNeXt(nn.Module):
    def __init__(self, in_channel, channel, h_channel, n_fft, hop_length, num_layers):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel, eps=1e-6)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel, eps=1e-6)
        self.out_conv = nn.Conv1d(channel, n_fft, 1)
        self.fc = nn.Conv1d(n_fft, hop_length, 1, bias=False)

        self.in_conv.apply(self._init_weights)
        self.layers.apply(self._init_weights)
        nn.init.trunc_normal_(self.out_conv.weight, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        o = self.out_conv(x)
        o = self.fc(o)
        o = o.reshape(o.size(0), 1, -1).clip(-1, 1)
        return o
