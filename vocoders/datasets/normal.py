import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


class NormalDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, segment_size):
        with open(file_path) as f:
            lines = f.readlines()
            data = [line.strip() for line in lines]
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.segment_size = segment_size

    def __getitem__(self, idx):
        bname = self.data[idx]

        wav, _ = torchaudio.load(self.wav_dir / f"{bname}.wav")
        if wav.shape[-1] > self.segment_size:
            s = random.randint(0, wav.shape[-1] - self.segment_size - 1)
            e = s + self.segment_size
            wav = wav[:, s:e]
        elif wav.shape[-1] < self.segment_size:
            wav = F.pad(wav, [0, self.segment_size - wav.shape[-1]])
        return bname, wav

    def __len__(self):
        return len(self.data)
