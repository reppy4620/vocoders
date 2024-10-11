import torch

from .normal import NormalLitModule


class WithCf0VuvLitModule(NormalLitModule):
    def forward(
        self, mel: torch.Tensor, cf0: torch.Tensor, vuv: torch.Tensor
    ) -> torch.Tensor:
        return self.net_g(mel, cf0, vuv).squeeze(1)

    def _process_generator(self, batch):
        _, y, cf0, vuv = batch
        y_mel = self.to_mel(y.squeeze(1))
        y_hat = self.net_g(y_mel, cf0, vuv)
        return y, y_hat
