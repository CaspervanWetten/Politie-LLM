import torch
import torch.nn as nn
from torch.nn import functional as F

from helpers import get_input
# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # X amount of tokens to predict the X+1th token
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 # embeddings in dimentions. N = dimensions!
n_head = 6 # Number of heads -> has to neatly divide n_embed
n_layer = 6 # Number of blocks
dropout = 0.2 # Dropout randomly drops some blocks during the training, meaning it strongly prevents overfitting
text = get_input("wouter")
# Bring down the layers and embedding dimensions to improve speed
#-----------------

	
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


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # A linear transformation of the output of the concationation
        return out
 

class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head # Should be 8 
        self.sa = MultiHeadAttention(n_head, head_size)  # sa = self-attention
        self.ffwd = FeedFoward(n_embed) # Makes the tokens thinks (self matrix multiplication)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        """Note that the sequence they appear is also the sequence they are used"""

		# Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        #We're not just encoding identity, we're also encoding position!
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # Final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # LM=loaded model
        # N_embed is the number of embedded dimentions
        # .Embedding creates a shape of vocab_size x vocab_size
        # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_em = self.token_embedding_table(idx)  # B,T,C
        pos_em = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_em + pos_em # B,T,C
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T, vocab_size)
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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # Does the prediction 
            logits = logits[:, -1, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, ook (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
        return idx
		

if __name__ == "__main__":
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
    state_dict = m.state_dict()
    with open("Code/state_dict.txt", "w") as f:
        for key, value in state_dict.items():
            f.write(f"{key} with accompanying {value}")
    torch.save(state_dict, "Code/models/statedict")
    torch.save(m, "Code/models/statedict")

