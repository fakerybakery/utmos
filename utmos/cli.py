import torch
import torch.nn as nn
from cached_path import cached_path
import click
from .score import Score
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

@click.command()
@click.argument('filename', type=click.Path(exists=True))
def main(filepath):
    model = Score()
    score = model.calculate_wav_file('audio_file.wav')
    print(f"Score: {score}")
if __name__ == '__main__':
    main()
