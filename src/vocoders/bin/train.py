from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from vocoders.utils.logging import logger


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    out_dir = Path(cfg.train.out_dir)
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        out_dir / "train.log",
        format="<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}",
        backtrace=True,
        diagnose=True,
    )

    OmegaConf.save(cfg, out_dir / "config.yaml")

    seed_everything(cfg.train.seed)

    lit_module = instantiate(cfg.lit_module, params=cfg, _recursive_=False)

    csv_logger = CSVLogger(save_dir=out_dir, name="logs/csv")
    tb_logger = TensorBoardLogger(save_dir=out_dir, name="logs/tensorboard")
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
        **cfg.train.trainer_args,
    )
    trainer.fit(
        model=lit_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
