import torch
import torch.nn as nn
from torch.nn import functional as F

from helpers import get_input
# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?
max_iters = 10000 # 10k
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 # 32 embeddings in dimentions. N = dimensions!
text = get_input("wouter")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stringtoint = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
inttostring = { i:ch for i,ch in enumerate(chars) }
decode = lambda intL: ''.join([inttostring[i] for i in intL]) # backup decoding algo
tokenizer_warning = lambda: print("WARNING: NO TOKENIZER SELECTED, USING FALLBACK")
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # Gebruik de eerste 90% als train data
train_data = data[:n]
val_data = data[n:]
#-----------------



class Transformer(nn.Module):
    def __init__(self, **kwarg):
        super().__init__()
        # Expected keyword arguments
        self.train_data = kwarg.get("train_data", train_data)
        self.val_data = kwarg.get("val_data", val_data)
        self.tokenizer = kwarg.get("tokenizer", tokenizer_warning())
        self.vocab_size = self.tokenizer.vocab_size if kwarg.get("tokenzier") != None else vocab_size
        self.chars = self.tokenizer.chars if kwarg.get("tokenzier") != None else chars
        [setattr(self, key, value) for key, value in kwarg.items()] # Arbitrarily accept all keywords passed

		# Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embed)

        """Note that the sequence they appear is also the sequence they are used"""

        #We're not just encoding identity, we're also encoding position!
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4) # sa = self-attention
        self.ffwd = FeedFoward(n_embed=n_embed) # Makes the tokens thinks (self matrix multiplication)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            nn.LayerNorm(n_embed)
        )
        
        self.lm_head = nn.Linear(n_embed, self.vocab_size) # LM=loaded model
         # N_embed is the number of embedded dimentions
         # .Embedding creates a shape of vocab_size x vocab_size
        # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table
                # Use this specific optimizier algorithm TODO understand this
        self.optimizer = torch.optim.AdamW(super().parameters(), lr=learning_rate)
       

    def encode(self, string):
        """
        Calls tokenizer.encode, else falls to general backup
        """
        if self.tokenzier:
            self.tokenizer.encode(string)
        else:
            return encode(string)

    def decode(self, string):
        """ 
        Calls tokenizer.decode, else falls to general backup
        """
        if self.tokenizer:
            self.tokenizer.decode(string)
        else:
            return decode(string)
        
    def forward(self, context, targets=None):
        B, T = context.shape
        #context and targets are both (B,T) tensor of integers
        tok_em = self.token_embedding_table(context)  # B,T,C
        pos_em = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_em + pos_em # B,T,C
        x = self.sa_heads(x) # Apply one head of self-attention B,T,C
        x = self.ffwd(x) # B,T,C
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
        
    def generate(self, context, max_new_tokens):
        """
        Requires a tensor as context input (encoded using the same tokenizer),
        max_new_tokens defaults to 512

        returns encoded tensor response
        """
    # context is (B,T) array of indices
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            context_cond = context[:, -block_size:]
            logits, loss = self(context_cond) # Does the prediction 
            logits = logits[:, -1, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, ook (B,C)
            context_next = torch.multinomial(probs, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
            context = torch.cat((context, context_next), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
        return context

    def optimize(self):
        for iter in range(max_iters):
            # Every once in a while evaluate on the loss on train and val sets
            if iter % eval_interval == 0:
                losses = self.estimate_loss()
                print(losses)
                print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

            # Sample a batch of data
            xb, yb = self._get_batch("train")

            # Evaluate the loss
            logits, loss = model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def _get_batch(self, split, data=None):
        # Generate a small batch of inputs x and targets y
        data = data if data != None else self.train_data if split =='train' else self.val_data 
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad() # A context manager (?) to tell PyTorch to not make these backwards callable, e.g. skip back propagation
    def estimate_loss(self):
        out = {}
        model.eval() # Set the model to eval mode
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self._get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train() # Set the model to training mode
        return out

# Basic helper modules, build an interface when needed
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v 
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out
class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):

    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        # This will be a slight deviation from "attention is all you need"
        # We will be doing pre-attention layer normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x 
		

if __name__ == "__main__": 
    model = Transformer() # Instantiate the transformer
    m = model.to(device) # Assign it to the device (CPU on NR2)
    model.optimize() # Optimize (i.e. train) the trainsformer
    

    context = torch.zeros((1, 1), dtype=torch.long, device=device) # Context is, basically, the input string
    print()
    print("AAAS",decode(m.generate(context)[0].tolist()))
 
    

