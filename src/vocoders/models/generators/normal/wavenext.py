import torch.nn as nn

from vocoders.layers.convnext import ConvNeXtLayer


class WaveNeXt(nn.Module):
    def __init__(self, in_channel, channel, h_channel, n_fft, hop_length, num_layers):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_linear = nn.Linear(channel, n_fft)
        self.fc = nn.Linear(n_fft, hop_length, bias=False)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2))
        o = self.out_linear(x)
        o = self.fc(o)
        o = o.reshape(o.size(0), 1, -1).clip(-1, 1)
        return o
