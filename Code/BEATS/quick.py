import torch

from Tokenizers import Tokenizers, TokenizersConfig

checkpoint = torch.load('Code\BEATS\Tokenizer_iter3_plus_AS2M.pt')

T = TokenizersConfig(checkpoint['cfg'])
BEATS = Tokenizers(T)
BEATS.load_state_dict(checkpoint['model'])

print(BEATS)