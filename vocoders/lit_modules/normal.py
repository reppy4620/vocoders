import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule

from vocoders.losses.gan import discriminator_loss, feature_loss, generator_loss


class NormalLitModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss_coef = params.train.loss_coef
        self.automatic_optimization = False

        self.net_g = instantiate(params.generator)
        self.net_d = instantiate(params.discriminator)

        self.to_mel = instantiate(params.mel)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net_g(mel).squeeze(1)

    def _process_generator(self, batch):
        _, y = batch
        y_mel = self.to_mel(y.squeeze(1))
        y_hat = self.net_g(y_mel)
        return y, y_mel, y_hat

    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        y, y_mel, y_hat = self._process_generator(batch)
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
        loss_fm = self.loss_coef.fm * feature_loss(fmap_real, fmap_fake)
        loss_g = loss_gen + loss_mel + loss_fm
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = {"disc": loss_d, "gen": loss_gen, "mel": loss_mel, "fm": loss_fm}

        self.log_dict(loss_dict, prog_bar=True)

    def training_step(self, batch):
        self._handle_batch(batch, train=True)

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        self._handle_batch(batch, train=False)

    def configure_optimizers(self):
        optimizer_g = instantiate(self.params.optimizer, params=self.net_g.parameters())
        optimizer_d = instantiate(self.params.optimizer, params=self.net_d.parameters())
        scheduler_g = instantiate(self.params.scheduler, optimizer=optimizer_g)
        scheduler_d = instantiate(self.params.scheduler, optimizer=optimizer_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
