from pathlib import Path

import hydra
import numpy as np
import pyworld as pw
import soundfile as sf
from joblib import Parallel, delayed
from vocoders.utils.tqdm import tqdm_joblib


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    wav_dir = Path(cfg.path.wav_dir)
    data_root = Path(cfg.preprocess.out_dir)
    [
        (data_root / d).mkdir(parents=True, exist_ok=True)
        for d in ["f0", "cf0", "vuv", "filelists"]
    ]

    print("1. Generating filelists")
    wav_files = list(sorted(wav_dir.glob("*.wav")))
    valid_size = int(len(wav_files) * 0.02)
    train_files = wav_files[valid_size:]
    valid_files = wav_files[:valid_size]
    with open(cfg.path.train_file, "w") as f:
        f.writelines([f"{wav_path.stem}\n" for wav_path in train_files])
    with open(cfg.path.valid_file, "w") as f:
        f.writelines([f"{wav_path.stem}\n" for wav_path in valid_files])

    print("2. Extracting f0, cf0, and vuv")

    def _process(wav_file):
        bname = wav_file.stem
        wav, sr = sf.read(wav_file)
        assert sr == cfg.mel.sample_rate
        f0, _ = pw.harvest(
            wav, sr, frame_period=cfg.mel.hop_length / cfg.mel.sample_rate * 1e3
        )
        vuv = (f0 != 0).astype(np.float32)
        x = np.arange(len(f0))
        idx = np.nonzero(f0)
        cf0 = np.interp(x, x[idx], f0[idx])
        np.save(data_root / f"f0/{bname}.npy", f0)
        np.save(data_root / f"cf0/{bname}.npy", cf0)
        np.save(data_root / f"vuv/{bname}.npy", vuv)

    with tqdm_joblib(len(wav_files)):
        Parallel(n_jobs=8)(delayed(_process)(f) for f in wav_files)


if __name__ == "__main__":
    main()
