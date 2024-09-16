# Generic Police Transformer 
# Stupid name, it's explicitly not generic but shhhhh


with open("Code/datasets/hamlet.txt") as f:
    text = f.read()
chars = sorted(list(set(text)))

vocab_len = len(chars)

print(enumerate(chars))

test = "test"


"""Slechts een manier van tokenizing (wat gwn encoding is), een andere is tiktoken (GPT) of Sentencepiece(Google)"""
stoi = {ch:i for i,ch in enumerate(chars)} # Enumerate = dict met 'item':'locatie van item in string'; -> {}'E':0, 'n':1, 'u':2}
itos = {i:ch for i,ch in enumerate(chars)} # Verschil is volgorde -> 'e':0, {'t':1, 'a':2, 'r':3}
encode = lambda string: [stoi[c] for c in string] # Letterlijk een mapping van let-get gebasseerd op volgorde
decode = lambda integerl: ''.join([itos[i] for i in integerl])


print(encode(test))
print(decode(encode(test)))


import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # De duizend eerste karakters

n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# We gaan niet de gehele input in de transformer flikkeren (want CPU kracht)
# Dus we gaan het eerst in kleine stukjes knippen, de max lengte is de block_size

block_size = 8
train_data[:block_size+1]
# Je doet +1, want de transformer voorspeld dan de hele lijn, en output de +1

x = train_data[:block_size]
y = train_data[1:block_size]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} target is {target}")
    # Dit traint de transformer ook om gewend te zijn aan de variabele lengtes plus haar voorspelling




