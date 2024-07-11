import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torchaudio
from hydra.utils import get_class, instantiate
from pesq import pesq
from torchaudio.functional import resample
from tqdm import tqdm


@hydra.main(config_path="conf", version_base=None, config_name="config")
@torch.inference_mode()
def main(cfg):
    out_dir = Path(cfg.syn.out_dir)
    [(out_dir / s).mkdir(parents=True, exist_ok=True) for s in ["wav"]]
    wav_dir = Path(cfg.path.wav_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module = (
        get_class(cfg.lit_module)
        .load_from_checkpoint(cfg.syn.ckpt_path, params=cfg)
        .to(device)
        .eval()
    )
    to_mel = instantiate(cfg.mel).to(device)

    mos_predictor = (
        torch.hub.load("tarepan/SpeechMOS:v1.0.0", "utmos22_strong", trust_repo=True)
        .to(device)
        .eval()
    )

    rtf_list = []
    mos_list = []
    pesq_list = []

    with open(cfg.file_path, mode="r") as f:
        bnames = [line.strip() for line in f.readlines()]

    for bname in tqdm(bnames, total=len(bnames)):
        wav_path = wav_dir / f"{bname}.wav"
        wav, sr = torchaudio.load(wav_path)
        assert sr == cfg.mel.sample_rate
        wav = wav.to(device)
        mel = to_mel(wav)
        torch.cuda.synchronize()
        # predict wav from mel
        s = time.time()
        o = lit_module(mel)
        torch.cuda.synchronize()
        rtf = (time.time() - s) / o.shape[-1] * sr
        rtf_list.append(rtf)
        torchaudio.save(
            out_dir / f"wav/{bname}.wav",
            o.cpu(),
            sr,
        )

        # calculate UTMOS
        gt_mos = mos_predictor(wav, sr)
        syn_mos = mos_predictor(o, sr)
        mos_list.append((gt_mos.item(), syn_mos.item()))

        # calculate PESQ
        wav = resample(wav, sr, 16000)
        o = resample(o, sr, 16000)
        pesq_score = pesq(16000, wav.squeeze().cpu().numpy(), o.squeeze().cpu().numpy())
        pesq_list.append(pesq_score)

    rtf = np.mean(rtf_list)
    gt_mos, syn_mos = map(np.mean, zip(*mos_list))
    pesq_score = np.mean(pesq_list)
    with open(out_dir / "result.txt", "w") as f:
        f.write(f"RTF: {rtf} (calculated by {len(bnames)} files)\n")
        f.write(f"UTMOS-GT: {gt_mos} (calculated by {len(bnames)} files)\n")
        f.write(f"UTMOS-Syn: {syn_mos} (calculated by {len(bnames)} files)\n")
        f.write(f"PESQ: {pesq_score} (calculated by {len(bnames)} files)\n")


if __name__ == "__main__":
    main()
