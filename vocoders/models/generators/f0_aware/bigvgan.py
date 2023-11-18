import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm

from vocoders.layers.activations import AntiAliasActivation
from vocoders.layers.amp import AMPBlock
from vocoders.layers.nsf import SourceModuleHnNSF


class F0AwareBigVGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        sample_rate,
        harmonic_num,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=u // 2 + u % 2,
                        output_padding=u % 2,
                    )
                )
            )
            if i < len(upsample_rates) - 1:
                stride_f0 = np.prod(upsample_rates[i + 1 :])  # noqa
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(
                    nn.Conv1d(
                        1, upsample_initial_channel // (2 ** (i + 1)), kernel_size=1
                    )
                )

        self.amps = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.amps.append(
                nn.ModuleList(
                    [
                        AMPBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.act_post = AntiAliasActivation(channel)
        self.conv_post = weight_norm(
            nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )

        self.f0_upsample = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

    def forward(self, x, f0):
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)

        x = self.conv_pre(x)
        for up, amp, noise_conv in zip(self.upsamples, self.amps, self.noise_convs):
            x = up(x)
            x_source = noise_conv(har_source)
            x = x + x_source
            xs = 0
            for layer in amp:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for amp in self.amps:
            amp.remove_weight_norm()
        remove_weight_norm(self.conv_post)
