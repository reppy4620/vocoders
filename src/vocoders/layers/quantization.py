import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 最近傍整数に丸める
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        # 入力と同じ勾配を返す
        return grad_outputs


round_ste = StraightThroughEstimator.apply
eps = 1e-8


class QuantizedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        elementwise_affine=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        # self.layer_norm = nn.LayerNorm(
        #     in_channels, elementwise_affine=elementwise_affine
        # )

        self.p_precision = 8
        self.Qp = 2 ** (self.p_precision - 1)  # 128.0

    def forward(self, x):
        # x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)
        w = self.weight
        beta = torch.mean(torch.abs(w))
        w_quantized = round_ste(torch.clamp(w / (beta + eps), -1, 1))
        gamma = torch.max(torch.abs(x))
        x_scaled = (x * self.Qp) / (gamma + eps)
        x_quantized = torch.clamp(x_scaled, -self.Qp, self.Qp - 1)
        y_quantized = F.conv1d(
            x_quantized,
            w_quantized,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        y = (y_quantized * beta * gamma) / self.Qp
        return y

    def remove_weight_norm(self):
        remove_parametrizations(self, "weight")


class QuantizedConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
        elementwise_affine=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=False,
            dilation=dilation,
        )

        self.layer_norm = nn.LayerNorm(
            in_channels, elementwise_affine=elementwise_affine
        )

        self.p_precision = 8
        self.Qp = 2 ** (self.p_precision - 1)

    def forward(self, x):
        # x_norm = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x_norm = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + eps)
        w = self.weight
        beta = torch.mean(torch.abs(w))
        w_quantized = round_ste(torch.clamp(w / (beta + eps), -1, 1))
        gamma = torch.max(torch.abs(x_norm))
        x_scaled = (x_norm * self.Qp) / (gamma + eps)
        x_quantized = torch.clamp(x_scaled, -self.Qp, self.Qp - 1)
        y_quantized = F.conv_transpose1d(
            x_quantized,
            w_quantized,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        y = (y_quantized * beta * gamma) / self.Qp

        return y

    def remove_weight_norm(self):
        remove_parametrizations(self, "weight")
