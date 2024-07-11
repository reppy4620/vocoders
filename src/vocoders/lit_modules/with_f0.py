import torch

from .normal import NormalLitModule


class WithF0LitModule(NormalLitModule):
    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        return self.net_g(mel, f0).squeeze(1)

    def _process_generator(self, batch):
        _, y, f0 = batch
        y_mel = self.to_mel(y.squeeze(1))
        y_hat = self.net_g(y_mel, f0)
        return y, y_hat
