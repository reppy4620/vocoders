import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from vocoders.layers.mrf import PReLUMRFBlock
from vocoders.layers.nsf import SignalGenerator


class PReLUVocoder(nn.Module):
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
    ):
        super().__init__()
        assert hop_length == np.prod(upsample_rates)
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )

        self.acts = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.acts.append(nn.PReLU(upsample_initial_channel // (2**i)))
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

        self.mrfs = nn.ModuleList()
        for i in range(len(upsample_rates)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        PReLUMRFBlock(channel, kernel_size=k, dilations=ds)
                        for k, ds in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )

        self.act_post = nn.PReLU(upsample_initial_channel // 16)
        self.conv_post = weight_norm(nn.Conv1d(channel, 1, 7, 1, padding=3))

        self.downs = nn.ModuleList(
            [
                nn.Identity(),
                nn.AvgPool1d(upsample_rates[-1], upsample_rates[-1]),
                nn.AvgPool1d(upsample_rates[-2], upsample_rates[-2]),
                nn.Identity(),
            ]
        )
        self.mrf_pre_convs = nn.ModuleList(
            [
                nn.Identity(),
                weight_norm(
                    nn.Conv1d(
                        upsample_initial_channel // 4 + 3,
                        upsample_initial_channel // 4,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        upsample_initial_channel // 8 + 3,
                        upsample_initial_channel // 8,
                        kernel_size=3,
                        padding=1,
                    )
                ),
                nn.Identity(),
            ]
        )
        self.upsample = nn.Upsample(scale_factor=hop_length)
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate, hop_length=hop_length
        )

    def adjust_cat(self, x, sine, vuv, noise):
        t_x, t_s = x.size(-1), sine.size(-1)
        if t_x < t_s:
            x = torch.cat([x, sine[..., :t_x], vuv[..., :t_x], noise[..., :t_x]], dim=1)
        else:
            x = torch.cat([x[..., :t_s], sine, vuv, noise], dim=1)
        return x

    def forward(self, x, cf0, vuv):
        cf0_up, vuv_up = self.upsample(cf0[:, None, :]), self.upsample(vuv[:, None, :])
        sine, vuv, _ = self.signal_generator(cf0_up, vuv_up)

        sources = []
        for down in self.downs:
            sine = down(sine)
            vuv = down(vuv)
            noise = torch.randn_like(vuv)
            sources.insert(0, (sine, vuv, noise))

        x = self.conv_pre(x)
        for i, (act, up, pre, mrf) in enumerate(
            zip(self.acts, self.upsamples, self.mrf_pre_convs, self.mrfs)
        ):
            x = act(x)
            x = up(x)
            if i == 1 or i == 2:
                sine, vuv, noise = sources[i]
                x = self.adjust_cat(x, sine, vuv, noise)
                x = pre(x)
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)
        x = x.tanh()
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for conv in self.res_conv_pre:
            if not isinstance(conv, nn.Identity):
                remove_weight_norm(conv)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
