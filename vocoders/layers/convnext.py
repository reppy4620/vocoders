import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtLayer(nn.Module):
    def __init__(self, channel, h_channel, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channel, channel, kernel_size=7, padding=3, groups=channel
        )
        self.norm = nn.LayerNorm(channel)
        self.pw_conv1 = nn.Linear(channel, h_channel)
        self.pw_conv2 = nn.Linear(h_channel, channel)
        self.scale = nn.Parameter(
            torch.full(size=(channel,), fill_value=scale), requires_grad=True
        )

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = x.transpose(1, 2)
        x = res + x
        return x
