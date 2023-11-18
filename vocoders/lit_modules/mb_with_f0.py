import torch
import torch.nn.functional as F

from vocoders.layers.pqmf import PQMF
from vocoders.losses.gan import discriminator_loss, feature_loss, generator_loss
from vocoders.losses.stft import MultiResolutionSTFTLoss

from .normal import NormalLitModule


class MBWithF0LitModule(NormalLitModule):
    def __init__(self, params):
        super().__init__(params=params)
        self.stft_loss = MultiResolutionSTFTLoss()
        self.pqmf = PQMF()

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        o, _ = self.net_g(mel, f0)
        return o.squeeze(1)

    def _process_generator(self, batch):
        _, y, f0 = batch
        y_m = self.pqmf.analysis(y)
        y_mel = self.to_mel(y.squeeze(1))
        y_hat, y_hat_m = self.net_g(y_mel, f0)
        return y, y_m, y_mel, y_hat, y_hat_m

    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        y, y_m, y_mel, y_hat, y_hat_m = self._process_generator(batch)
        y_hat_mel = self.to_mel(y_hat.squeeze(1))

        d_real, d_fake, _, _ = self.net_d(y, y_hat.detach())
        loss_d = discriminator_loss(d_real, d_fake)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.net_d(y, y_hat)
        loss_gen = generator_loss(d_fake)
        loss_mel = self.loss_coef.mel * F.l1_loss(y_hat_mel, y_mel)
        loss_stft = self.stft_loss(y_hat_m, y_m)
        loss_fm = self.loss_coef.fm * feature_loss(fmap_real, fmap_fake)
        loss_g = loss_gen + loss_mel + loss_stft + loss_fm
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = {
            "disc": loss_d,
            "gen": loss_gen,
            "mel": loss_mel,
            "stft": loss_stft,
            "fm": loss_fm,
        }

        self.log_dict(loss_dict, prog_bar=True)
