import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class MelSpectrogramTransform(MelSpectrogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_length = (self.n_fft - self.hop_length) // 2

    def to_spec(self, wav):
        wav = F.pad(wav, [self.pad_length, self.pad_length], mode="reflect")
        spec = self.spectrogram(wav)
        return spec

    def to_mel(self, wav):
        spec = self.to_spec(wav)
        mel = self.spec_to_mel(spec)
        return mel

    def spec_to_mel(self, spec):
        mel = self.mel_scale(spec)
        mel = torch.log(torch.clamp_min(mel, min=1e-5))
        return mel

    def forward(self, wav):
        return self.to_mel(wav)
