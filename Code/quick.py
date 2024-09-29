import torch
import torchaudio
import torch.nn.functional as F
import time 
from Tokenizers.Audio import AudioTokenizer
from GPTInterface import Transformer
from Tokenizers.Text import TextTokenizer
from helpers import get_input


file_path = r'Code\datasets\audio\1-62565-A-44.wav'


text =  get_input("wouter")
# TT = TextTokenizer(data=text, debug=False)
AT = AudioTokenizer(debug=True, data=file_path)
T = Transformer(Tokenizer=AT)








# print(f"Before optimization:")
# context = "Ik heet Casper. Wie ben jij?" 
# generated = T.model.generate(context)
# decoded = T.model.decode(generated)
# print(f"context: {context}\nresulted in generation:\n{decoded}")

# start_time = time.time()
# T.optimize()
# end_time = time.time()
# print(f"Optimizing took exact: {end_time - start_time}")
# T.save_std("V6")

# print(f"Aftger optimization optimization:")
# context = "Ik heet Casper. Wie ben jij?" 
# generated = T.model.generate(context)
# decoded = T.model.decode(generated)
# print(f"context: {context}\nresulted in generation:\n{decoded}")


# encoded = AT.encode(file_path)
# print(f"Encoded .wav: {encoded}")

# print(T.model.generate(encoded))



# dims = 2
# length = 16

# a = torch.tensor([[0,1,2,3],[2,3,4,5]])
# b = torch.tensor([[21,23,45,28, 33],[8,9,10,12,25]])
# while a.size(dims-1) < length:
#     a = F.pad(a, (0, 1), "constant", 0)

# while b.size(dims-1) < length:
#     b = F.pad(b, (0, 1), "constant", 0)

# print(torch.cat((a, b), dim=0))






