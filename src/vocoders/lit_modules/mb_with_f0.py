import torch
from hydra.utils import instantiate
from vocoders.layers.pqmf import PQMF

from .normal import NormalLitModule


class MBWithF0LitModule(NormalLitModule):
    def __init__(self, params):
        super().__init__(params=params)
        self.pqmf = PQMF()

        self.stft_loss = instantiate(params.loss.gen.stft)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        o, _ = self.net_g(mel, f0)
        return o.squeeze(1)

    def _process_generator(self, batch):
        _, y, f0 = batch
        y_m = self.pqmf.analysis(y)
        y_mel = self.to_mel(y.squeeze(1))
        y_hat, y_hat_m = self.net_g(y_mel, f0)
        return y, y_m, y_hat, y_hat_m

    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        y, y_m, y_hat, y_hat_m = self._process_generator(batch)

        d_real, d_fake, _, _ = self.net_d(y, y_hat.detach())
        with torch.autocast(device_type="cuda", enabled=False):
            loss_d = self.disc_loss(d_real, d_fake, self.loss_coef.second_disc)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.net_d(y, y_hat)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_gen = self.gen_loss(d_fake)
            loss_mel = self.mel_loss(y_hat, y)
            loss_stft = self.stft_loss(y_hat_m, y_m)
            loss_fm = self.fm_loss(fmap_real, fmap_fake)
            loss_g = (
                loss_gen
                + self.loss_coef.mel * loss_mel
                + loss_stft
                + self.loss_coef.fm * loss_fm
            )
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = dict(
            disc=loss_d,
            gen=loss_gen,
            mel=loss_mel,
            stft=loss_stft,
            fm=loss_fm,
        )

        self.log_dict(loss_dict, prog_bar=True)
