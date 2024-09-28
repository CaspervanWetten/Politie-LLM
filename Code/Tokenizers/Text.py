import torch

from Tokenizers.Generic import GenericTokenizer


class TextTokenizer (GenericTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distinct_tokens = sorted(list(set(self.data)))
        self.num_distinct_tokens = len(self.distinct_tokens)
        self.stringtoint = { ch:i for i,ch in enumerate(self.distinct_tokens) } # {"A":0, "B":1, ..., "!": 80}
        self.inttostring= { i:ch for i,ch in enumerate(self.distinct_tokens) } # {0:"A", 1:"B", ..., 80:"!"}

    def encode(self, input: str) -> list:
        return [self.stringtoint[c] for c in input] # MVP encoding algo
    
    def decode(self, input: list) -> str:
        return ''.join([self.inttostring[i] for i in input]) # MVP encoding algo
    
    def pad_input(self, input: torch.Tensor, mask_length: int=16, dims: int=2) -> torch.Tensor:
        """
        Default naar 16 mask length en 2 dims ()
        """
        return super().pad_input(input, mask_length, dims)


        
    