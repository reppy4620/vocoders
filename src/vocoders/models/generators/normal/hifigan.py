import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from vocoders.layers.mrf import MRFBlock
from vocoders.layers.quantization import QuantizedConv1d, QuantizedConvTranspose1d
from vocoders.utils.const import LRELU_SLOPE


class HiFiGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = nn.ModuleList()
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

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        MRFBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.conv_post = weight_norm(
            nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        x = self.conv_pre(x)
        for up, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.mrfs:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class BitHiFiGAN(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.conv_pre = weight_norm(
            QuantizedConv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    QuantizedConvTranspose1d(
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
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        MRFBlock(
                            channel,
                            kernel_size=k,
                            dilations=d,
                            conv_layer=QuantizedConv1d,
                        )
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.conv_post = weight_norm(
            nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x):
        x = self.conv_pre(x)
        for up, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        self.conv_pre.remove_weight_norm()
        for up in self.upsamples:
            up.remove_weight_norm()
        for mrf in self.mrfs:
            for block in mrf:
                block.remove_weight_norm()
        remove_parametrizations(self.conv_post, "weight")
