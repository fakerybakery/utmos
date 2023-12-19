import lightning_module
import torch
import torchaudio
from cached_path import cached_path
import unittest

class Score:
    """Predicting score for each audio clip."""

    def __init__(
        self,
        ckpt_path: str = cached_path('hf://ttseval/utmos/model.ckpt'),
        input_sample_rate: int = 16000,
        device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained checkpoint of UTMOS strong learner.
            input_sample_rate: sampling rate of input audio tensor. The input audio tensor
                is automatically downsampled to 16kHz.
        """
        print(f"Using device: {device}")
        self.device = device
        self.model = lightning_module.BaselineLightningModule.load_from_checkpoint(
            ckpt_path).eval().to(device)
        self.in_sr = input_sample_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        ).to(device)
    
    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3. 
        """
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError('Dimension of input tensor needs to be <= 3.')
        if self.in_sr != 16000:
            out_wavs = self.resampler(out_wavs)
        bs = out_wavs.shape[0]
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
            'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
        }
        with torch.no_grad():
            output = self.model(batch)
        
        return output.mean(dim=1).squeeze(1).cpu().detach().numpy()*2 + 3


class TestFunc(unittest.TestCase):
    """Test class."""

    def test_1dim_0(self):
        scorer = Score(input_sample_rate=16000)
        seq_len = 10000
        inp_audio = torch.ones(seq_len)
        pred = scorer.score(inp_audio)
        self.assertGreaterEqual(pred, 0.)
        self.assertLessEqual(pred, 5.)

    def test_1dim_1(self):
        scorer = Score(input_sample_rate=24000)
        seq_len = 10000
        inp_audio = torch.ones(seq_len)
        pred = scorer.score(inp_audio)
        self.assertGreaterEqual(pred, 0.)
        self.assertLessEqual(pred, 5.)

    def test_2dim_0(self):
        scorer = Score(input_sample_rate=16000)
        seq_len = 10000
        inp_audio = torch.ones(1, seq_len)
        pred = scorer.score(inp_audio)
        self.assertGreaterEqual(pred, 0.)
        self.assertLessEqual(pred, 5.)

    def test_2dim_1(self):
        scorer = Score(input_sample_rate=24000)
        seq_len = 10000
        inp_audio = torch.ones(1, seq_len)
        pred = scorer.score(inp_audio)
        print(pred)
        print(pred.shape)
        self.assertGreaterEqual(pred, 0.)
        self.assertLessEqual(pred, 5.)

    def test_3dim_0(self):
        scorer = Score(input_sample_rate=16000)
        seq_len = 10000
        batch = 8
        inp_audio = torch.ones(batch, 1, seq_len)
        pred = scorer.score(inp_audio)
        for p in pred:
            self.assertGreaterEqual(p, 0.)
            self.assertLessEqual(p, 5.)

    def test_3dim_1(self):
        scorer = Score(input_sample_rate=24000)
        seq_len = 10000
        batch = 8
        inp_audio = torch.ones(batch, 1, seq_len)
        pred = scorer.score(inp_audio)
        for p in pred:
            self.assertGreaterEqual(p, 0.)
            self.assertLessEqual(p, 5.)

if __name__ == '__main__':
    unittest.main()