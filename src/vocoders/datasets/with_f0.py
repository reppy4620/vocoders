import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class WithF0Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, f0_dir, frame_segment_size, hop_length):
        with open(file_path) as f:
            lines = f.readlines()
            data = [line.strip() for line in lines]
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.f0_dir = Path(f0_dir)
        self.frame_segment_size = frame_segment_size
        self.hop_length = hop_length

    def __getitem__(self, idx):
        bname = self.data[idx]
        wav, _ = torchaudio.load(self.wav_dir / f"{bname}.wav")
        f0 = torch.FloatTensor(np.load(self.f0_dir / f"{bname}.npy"))
        if f0.shape[-1] > self.frame_segment_size:
            s = random.randint(0, f0.shape[-1] - self.frame_segment_size - 1)
            e = s + self.frame_segment_size
            f0 = f0[s:e]
            wav = wav[..., s * self.hop_length : e * self.hop_length]  # noqa
        else:
            f0 = F.pad(f0, [0, self.frame_segment_size - f0.shape[-1]])
            wav = F.pad(
                wav, [0, self.hop_length * (self.frame_segment_size - f0.shape[-1])]
            )
        return bname, wav, f0

    def __len__(self):
        return len(self.data)
