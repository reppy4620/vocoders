import torch.nn as nn

from vocoders.layers.convnext import ConvNeXtLayer


class Vocos(nn.Module):
    def __init__(self, in_channel, channel, h_channel, out_channel, num_layers, istft):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_conv = nn.Conv1d(channel, out_channel, 1)
        self.istft = istft

    def forward(self, x):
        x = self.pad(x)
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o
