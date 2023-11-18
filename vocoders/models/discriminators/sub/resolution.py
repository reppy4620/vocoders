import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from vocoders.layers.san import SANConv2d

LRELU_SLOPE = 0.1


class DiscriminatorR(nn.Module):
    def __init__(self, resolution, mult=1, is_san=False):
        super().__init__()

        self.resolution = resolution
        assert (
            len(self.resolution) == 3
        ), "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.d_mult = mult

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        if is_san:
            self.conv_post = SANConv2d(
                int(32 * self.d_mult), 1, kernel_size=(3, 3), stride=1, padding=(1, 1)
            )
        else:
            self.conv_post = weight_norm(
                nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
            )

    def forward(self, x, is_san=False):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x, is_san=is_san)
        if is_san:
            x_fun, x_dir = x
            fmap.append(x_fun)
            x_fun = torch.flatten(x_fun, 1, -1)
            x_dir = torch.flatten(x_dir, 1, -1)
            x = [x_fun, x_dir]
        else:
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=[[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]],
        is_san=False,
    ):
        super().__init__()
        self.resolutions = resolutions
        assert (
            len(self.resolutions) == 3
        ), "MRD requires list of list with len=3, each element having a list with len=3. got {}".format(
            self.resolutions
        )
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution, is_san=is_san)
                for resolution in self.resolutions
            ]
        )

    def forward(self, y, y_hat, is_san=False):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, is_san=is_san)
            y_d_g, fmap_g = d(x=y_hat, is_san=is_san)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
