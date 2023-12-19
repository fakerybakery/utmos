import torchaudio
import torch
import torch.nn as nn
from cached_path import cached_path
from lightning_module import *
import click
class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = (torch.arange(new_length) * (self.input_rate / self.output_rate))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
        return output
class Score:
    def __init__(self):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        if torch.backends.mps.is_available():
            device = 'mps'
        self.model = BaselineLightningModule.load_from_checkpoint(cached_path('hf://ttseval/utmos/model.ckpt')).eval().to(device)
    def calculate_wav_file(self, file):
        wav, sr = torchaudio.load(file)
        return self.calculate_wav(wav, sr)
    def calculate_wav(self, wav, sr):
        osr = 16_000
        batch = wav.unsqueeze(0).repeat(10, 1, 1)
        csr = ChangeSampleRate(sr, osr)
        out_wavs = csr(wav)
        batch = {
            'wav': out_wavs,
            'domains': torch.tensor([0]),
            'judge_id': torch.tensor([288])
        }
        with torch.no_grad():
            output = self.model(batch)
        return output.mean(dim=1).squeeze().detach().numpy() * 2 + 3
