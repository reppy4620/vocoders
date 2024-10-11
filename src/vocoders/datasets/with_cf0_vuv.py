import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class WithCf0VuvDataset(torch.utils.data.Dataset):
    def __init__(
        self, file_path, wav_dir, cf0_dir, vuv_dir, frame_segment_size, hop_length
    ):
        with open(file_path) as f:
            lines = f.readlines()
            data = [line.strip() for line in lines]
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.cf0_dir = Path(cf0_dir)
        self.vuv_dir = Path(vuv_dir)
        self.frame_segment_size = frame_segment_size
        self.hop_length = hop_length

    def __getitem__(self, idx):
        bname = self.data[idx]
        wav, _ = torchaudio.load(self.wav_dir / f"{bname}.wav")
        cf0 = torch.FloatTensor(np.load(self.cf0_dir / f"{bname}.npy"))
        vuv = torch.FloatTensor(np.load(self.vuv_dir / f"{bname}.npy"))
        assert cf0.shape == vuv.shape, f"cf0 {cf0.shape} != vuv {vuv.shape}"
        if cf0.shape[-1] > self.frame_segment_size:
            s = random.randint(0, cf0.shape[-1] - self.frame_segment_size - 1)
            e = s + self.frame_segment_size
            cf0 = cf0[s:e]
            vuv = vuv[s:e]
            wav = wav[..., s * self.hop_length : e * self.hop_length]  # noqa
        else:
            cf0 = F.pad(cf0, [0, self.frame_segment_size - cf0.shape[-1]])
            vuv = F.pad(vuv, [0, self.frame_segment_size - vuv.shape[-1]])
            wav = F.pad(
                wav, [0, self.hop_length * (self.frame_segment_size - cf0.shape[-1])]
            )
        return bname, wav, cf0, vuv

    def __len__(self):
        return len(self.data)
