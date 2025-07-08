import torch
import torch.nn as nn
import torch.nn.functional as F

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
        o = self.fc(o).transpose(1, 2)
        o = o.reshape(o.size(0), 1, -1).clip(-1, 1)
        return o


class WaveNeXtOverlap(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        h_channel,
        n_fft,
        hop_length,
        num_layers,
        overlap_kernel_size,
    ):
        super().__init__()
        self.overlap_kernel_size = overlap_kernel_size
        self.hop_length = hop_length

        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel)
        scale = 1 / num_layers
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel, scale) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_conv = nn.Conv1d(channel, n_fft, 1)

        self.overlap_conv = nn.Conv1d(
            n_fft,
            hop_length,
            kernel_size=overlap_kernel_size,
            padding=overlap_kernel_size - 1,
            bias=False,
        )

    def overlap_add(self, x):
        o = self.overlap_conv(x)
        _, C, L = o.shape
        pad_size = C * self.overlap_kernel_size - L + self.overlap_kernel_size - 1
        o = F.pad(o, [0, pad_size], mode="constant", value=0)
        shifts = torch.arange(C, device=x.device).reshape(1, C, 1) * (
            C * self.overlap_kernel_size - 1
        )
        range_idx = torch.arange(L, device=x.device).reshape(1, 1, L)
        indices = (range_idx + shifts) % L
        o = torch.gather(o, dim=2, index=indices)
        o = o.sum(dim=1, keepdim=True)[
            ..., self.overlap_kernel_size // 2 : -self.overlap_kernel_size // 2 + 1
        ]
        return o

    def forward(self, x):
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        o = self.out_conv(x)
        o = self.overlap_add(o)
        return o
