from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_df = pd.read_csv(args.loss_file)
    train_loss = (
        loss_df.groupby("epoch").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    )
    train_loss = (
        train_loss.groupby("epoch")
        .agg("mean")
        .reset_index(drop=True)
        .drop(["step"], axis=1)
    )
    valid_loss = (
        loss_df.groupby("epoch")
        .tail(1)
        .reset_index(drop=True)
        .drop(["epoch", "step"], axis=1)
    )

    epochs = list(range(1, train_loss.shape[0] + 1))
    for loss_name in tqdm(train_loss.columns):
        plt.figure(figsize=(12, 8))
        length = min(train_loss.shape[0], valid_loss.shape[0])
        plt.plot(epochs, train_loss[loss_name][:length], label="train")
        plt.plot(epochs, valid_loss[loss_name][:length], label="valid")
        plt.grid()
        plt.legend()
        plt.title(f"{loss_name}")
        plt.savefig(out_dir / f"{loss_name}.png")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loss_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
