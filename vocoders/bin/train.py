from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


@hydra.main(config_path="conf", version_base=None, config_name="train")
def main(cfg):
    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, out_dir / "config.yaml")

    seed_everything(cfg.train.seed)

    train_ds = instantiate(cfg.dataset.train)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    valid_ds = instantiate(cfg.dataset.valid)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    lit_module = instantiate(cfg.lit_module, params=cfg, _recursive_=False)

    csv_logger = CSVLogger(save_dir=out_dir, name="logs/csv", version=1)
    tb_logger = TensorBoardLogger(save_dir=out_dir, name="logs/tensorboard", version=1)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=cfg.train.save_ckpt_interval,
        save_last=True,
        save_top_k=-1,
    )
    ckpt_path = ckpt_dir / "last.ckpt" if (ckpt_dir / "last.ckpt").exists() else None

    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        max_steps=cfg.train.num_steps,
        callbacks=[ckpt_callback],
    )
    trainer.fit(
        model=lit_module,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
