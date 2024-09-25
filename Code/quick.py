from helpers import get_input
# Pretend that these are imported from the tokenizer
text = get_input("wouter")
chars = sorted(list(set(text)))
stringtoint = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
inttostring = { i:ch for i,ch in enumerate(chars) }
decode = lambda intL: ''.join([inttostring[i] for i in intL]) # backup decoding algo

from GPT import Transformer

tranny = Transformer()
tranny.optimize()

context = "Wat is uw naam?"
encoded = encode(context)
print(context)
print()
print(decode(tranny.generate(context=encoded)[0].tolist()))