import numpy as np
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import InverseSpectrogram
from vocoders.layers.activations import AntiAliasActivation
from vocoders.layers.amp import AMPBlock
from vocoders.layers.nsf import SourceModuleHnNSF
from vocoders.layers.pqmf import LearnablePQMF


class MSiSTFTF0AwareBigVGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
        sample_rate,
        hop_length,
        harmonic_num,
        istft_config,
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

            stride_f0 = np.prod(hop_length // np.prod(upsample_rates[: i + 1]))
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=stride_f0 * 2,
                    stride=stride_f0,
                    padding=stride_f0 // 2,
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

        self.f0_upsample = nn.Upsample(scale_factor=hop_length)
        self.m_source = SourceModuleHnNSF(sample_rate, harmonic_num)

        self.pqmf = LearnablePQMF()
        self.subbands = self.pqmf.subbands
        self.pad = nn.ReflectionPad1d([1, 0])
        self.istft = InverseSpectrogram(**istft_config)
        self.n_fft = self.istft.n_fft

        self.conv_post = weight_norm(
            nn.Conv1d(
                channel,
                self.subbands * (self.n_fft + 2),
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

    def forward(self, x, f0):
        f0 = self.f0_upsample(f0[:, None, :]).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)

        x = self.conv_pre(x)
        for i, (up, amp, noise_conv) in enumerate(
            zip(self.upsamples, self.amps, self.noise_convs)
        ):
            x = up(x)
            x_source = noise_conv(har_source)
            x = x + x_source
            if i == len(self.upsamples) - 1:
                x = self.pad(x)
            xs = 0
            for layer in amp:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)

        B, C, T = x.shape
        x = x.reshape(B, self.subbands, C // self.subbands, T)
        mag, phase = x.split(self.n_fft // 2 + 1, dim=2)
        s = mag.exp() * (phase.cos() + 1j * phase.sin())
        y_mb = self.istft(s)
        y_mb = y_mb.reshape(B, self.subbands, -1)
        y = self.pqmf.synthesis(y_mb)
        return y, y_mb

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for amp in self.amps:
            amp.remove_weight_norm()
        remove_weight_norm(self.conv_post)
