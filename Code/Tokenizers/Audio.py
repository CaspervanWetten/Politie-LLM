# Possible sound things could be:
# Frequency (in Hz, determines wavelength)
# Intensite (db/power, also describable as the amplitude)
# Sample rate (resolution of audio)
import torch
import torchaudio.kaldi_io as ta_kaldi

from Tokenizers.Generic import GenericTokenizer

class AudioTokenizer(GenericTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, audio) -> torch.tensor:
        """
        Accepts an audio signal and returns a properly tokenized version
        Audio is pre-processed (normalized, etc)
        Audio is feature extracted (MFCC, CQT, melspectograms)
        Audio is Quantized (whatever that means)
        Batched?
        """



        pass




def preprocess(self, source: torch.Tensor, fbank_mean: float = 15.41663, fbank_std: float = 6.55582,
) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank
