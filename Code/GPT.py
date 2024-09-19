<<<<<<< HEAD
import torch
import torch.nn as nn
from torch.nn import functional as F

from helpers import get_input
# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32 # 32 embeddings in dimentions. N = dimensions!
text = get_input()
#----------------



torch.manual_seed(1337) # For reproduceability

	
# Create a vocabulair of all the unique characters that occur in this text.
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping from the characters=>integers
stringtoint = { ch:i for i,ch in enumerate(chars) }
inttostring= { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stringtoint[c] for c in s]
decode = lambda l: ''.join([inttostring[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # Gebruik de eerste 90% als train data
train_data = data[:n]
val_data = data[n:]

# Data loading:
def get_batch(split):
    # Generate a small batch of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad() # A context manager (?) to tell PyTorch to not make these backwards callable, e.g. skip back propagation
def estimate_loss():
    out = {}
    model.eval() # Set the model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set the model to training mode
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
		# Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        #We're not just encoding identity, we're also encoding position!
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size) # LM=loaded model
         # N_embed is the number of embedded dimentions
         # .Embedding creates a shape of vocab_size x vocab_size
        # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table
        
    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        tok_em = self.token_embedding_table(idx) 
        logits = self.lm_head(tok_em) # (B,T, vocab_size)
        # Not Logits, token embeddings
        
        # Creates a (Batch X Time X Channel) tensor
        # De 'logit' is dus de voorspelde volgende token (i.e. woord/whatever) zonder enige context.
        # Je wilt de loss nog evalueren
        
        if targets is None:
            loss = None
        else:
            # Dit werkt niet, de input van cross_entropy moet BxCxT zijn ipv BxTxC
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # enkel 2d array, conformt aan cross_entropy functie, zelfde moet gebeurten bij targets
            targets = targets.view(B*T) # Voor reden die ik niet snap is targets.view(-1) hetzelfde
            loss = F.cross_entropy(logits, targets) # Hoe goed voorspel je het volgende karakter gebasseerd op de logit?
            # De waarde van 
            
            # Loss verwacht je dat -ln(1/vocab_size) is (ln=natuurlijk logarithme)
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # Does the prediction 
            logits = logits[:, -1, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, ook (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
        return idx
		
model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer:
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # Learning rates could be much higher -> SUper high learning rates for specific the same?

for iter in range(max_iters):
    # Every once in a while evaluate on the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(losses)
        print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
=======
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




>>>>>>> 86b52dba2bc319c394fe7fefb4693416e82c6728
