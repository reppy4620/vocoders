import torch.nn.functional as F

from vocoders.losses.san import discriminator_loss, feature_loss, generator_loss

from .mb_with_f0 import MBWithF0LitModule


class SanMBWithF0LitModule(MBWithF0LitModule):
    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        y, y_m, y_mel, y_hat, y_hat_m = self._process_generator(batch)
        y_hat_mel = self.to_mel(y_hat.squeeze(1))

        d_real, d_fake, _, _ = self.net_d(y, y_hat.detach(), is_san=True)
        loss_d = discriminator_loss(d_real, d_fake)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.net_d(y, y_hat, is_san=False)
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
