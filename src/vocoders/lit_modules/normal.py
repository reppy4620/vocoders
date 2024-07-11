import torch
from hydra.utils import get_method, instantiate
from lightning import LightningModule
from torch.utils.data import DataLoader
from vocoders.utils.logging import logger


class NormalLitModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss_coef = params.train.loss_coef
        self.automatic_optimization = False

        self.net_g = instantiate(params.generator)
        self.net_d = instantiate(params.discriminator)

        logger.info(
            f"Generator: {sum(p.numel() for p in self.net_g.parameters() if p.requires_grad) / 1e6:.2f}M",
        )
        logger.info(
            f"Discriminator: {sum(p.numel() for p in self.net_d.parameters() if p.requires_grad) / 1e6:.2f}M",
        )

        self.to_mel = instantiate(params.mel)

        self.disc_loss = get_method(params.loss.disc)
        self.gen_loss = get_method(params.loss.gen.gan)
        self.mel_loss = instantiate(params.loss.gen.mel)
        self.fm_loss = get_method(params.loss.gen.fm)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net_g(mel).squeeze(1)

    def _process_generator(self, batch):
        _, y = batch
        y_mel = self.to_mel(y.squeeze(1))
        y_hat = self.net_g(y_mel)
        return y, y_hat

    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        y, y_hat = self._process_generator(batch)

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
            loss_fm = self.fm_loss(fmap_real, fmap_fake)
            loss_g = (
                loss_gen + self.loss_coef.mel * loss_mel + self.loss_coef.fm * loss_fm
            )
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = dict(disc=loss_d, gen=loss_gen, mel=loss_mel, fm=loss_fm)

        self.log_dict(loss_dict, prog_bar=True)

    def training_step(self, batch):
        self._handle_batch(batch, train=True)

    def on_train_epoch_end(self):
        for sch in self.lr_schedulers():
            sch.step()

    def validation_step(self, batch, batch_idx):
        self._handle_batch(batch, train=False)

    def train_dataloader(self):
        ds = instantiate(self.params.dataset.train)
        dl = DataLoader(
            ds,
            batch_size=self.params.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.params.train.num_workers,
            pin_memory=True,
        )
        return dl

    def val_dataloader(self):
        ds = instantiate(self.params.dataset.valid)
        dl = DataLoader(
            ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=self.params.train.num_workers,
        )
        return dl

    def configure_optimizers(self):
        optimizer_g = instantiate(self.params.optimizer, params=self.net_g.parameters())
        optimizer_d = instantiate(self.params.optimizer, params=self.net_d.parameters())
        scheduler_g = instantiate(self.params.scheduler, optimizer=optimizer_g)
        scheduler_d = instantiate(self.params.scheduler, optimizer=optimizer_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


class WaveNeXtLitModule(NormalLitModule):
    def on_train_epoch_end(self):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for sch in self.lr_schedulers():
            sch.step()
