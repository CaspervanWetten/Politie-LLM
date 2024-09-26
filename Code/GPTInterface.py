import torch
import torch.nn as nn
from torch.nn import functional as F

from helpers import get_input

class Transformer():
    def __init__(self, **kwargs):
        """
        Accepts batch_size=32, block_size=8, max_iters=600
        eval_interval=300, learning_rate=1e-3, device=cuda if available else cpu
        eval_iters=200, n_embed=384,
        
        train_data,val_data and tokenizer (all these have fallback options)
        """
        # Passable hyperparameters
        self.batch_size = kwargs.get("batch_size", 32) # how many independent sequences will we process in parallel?
        self.block_size = 8 # What is the maximum context length for predictions?
        self.max_iters = 1200 # 10k
        self.eval_interval = 300
        self.learning_rate = 1e-3
        self.eval_iters = 200
        self.n_embed = 384 # 32 embeddings in dimentions. N = dimensions!

        # Arbitrarily accept all keywords passed
        [setattr(self, key, value) for key, value in kwargs.items()] 

        # Non-passable arguments
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialization of the model TODO What if one wants to load a pretrained model?
        self.model = self.Model(self)
        self.model.to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def optimize(self):
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(losses)
                print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")
            xb, yb = self.model._get_batch(split="train")
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad() # A context manager (?) to tell PyTorch to not make these backwards callable, e.g. skip back propagation
    def estimate_loss(self):
        out = {}
        self.model.eval() # Set the model to eval mode
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.model._get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train() # Set the model to training mode
        return out

    class Model(nn.Module):
        def __init__(self, transformer, tokenizer=None, data=None):
            super().__init__()
            #----------------- Fallback for lack of tokenizer
            # This is basically a MVP tokenizer
            text = get_input("wouter")
            chars = sorted(list(set(text)))
            vocab_size = len(chars)
            stringtoint = { ch:i for i,ch in enumerate(chars) }
            inttostring = { i:ch for i,ch in enumerate(chars) }
            self._bencode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
            self._bdecode = lambda intL: ''.join([inttostring[i] for i in intL]) # backup decoding algo
            tokenizer_warning = lambda: print("WARNING: NO TOKENIZER PASSED, USING FALLBACK")
            #-----------------
            
            # Tokenizer
            self.tokenizer = tokenizer if tokenizer != None else tokenizer_warning()
            self.vocab_size = self.tokenizer.vocab_size if tokenizer != None else vocab_size
            self.chars = self.tokenizer.chars if tokenizer != None else chars

            data = data if data != None else torch.tensor(self.encode(text), dtype=torch.long)
            n = int(0.9*len(data))
            self.train_data = data[:n]
            self.val_data = data[n:]

            # Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
            self.token_embedding_table = nn.Embedding(self.vocab_size, transformer.n_embed)

            """Note that the sequence they appear is also the sequence they are used"""

            #We're not just encoding identity, we're also encoding position!
            self.position_embedding_table = nn.Embedding(transformer.block_size, transformer.n_embed)
            self.sa_heads = transformer.MultiHeadAttention(transformer, 4, transformer.n_embed//4) # sa = self-attention
            self.ffwd = transformer.FeedFoward(n_embed=transformer.n_embed) # Makes the tokens thinks (self matrix multiplication)
            self.blocks = nn.Sequential(
                transformer.Block(transformer, transformer.n_embed, n_head=4),
                transformer.Block(transformer, transformer.n_embed, n_head=4),
                transformer.Block(transformer, transformer.n_embed, n_head=4),
                nn.LayerNorm(transformer.n_embed)
            )
            
            self.lm_head = nn.Linear(transformer.n_embed, self.vocab_size) # LM=loaded model
            # N_embed is the number of embedded dimentions
            # .Embedding creates a shape of vocab_size x vocab_size
            # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table
            
        def encode(self, string):
            """
            Calls tokenizer.encode, else falls to general backup
            """
            if self.tokenizer:
                self.tokenizer.encode(string)
            else:
                return self._bencode(string)

        def decode(self, string):
            """ 
            Calls tokenizer.decode, else falls to general backup
            """
            if self.tokenizer:
                self.tokenizer.decode(string)
            else:
                return self._bdecode(string)       

        def forward(self, context, targets=None):
            B, T = context.shape
            #context and targets are both (B,T) tensor of integers
            tok_em = self.token_embedding_table(context)  # B,T,C
            pos_em = self.position_embedding_table(torch.arange(T, device=transformer.device)) # T, C
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
        
        def generate(self, context, max_new_tokens=512):
            """
            Requires a tensor as context input (encoded using the same tokenizer),
            max_new_tokens defaults to 512

            returns encoded tensor response
            """
            if type(context) == str:
                context = torch.tensor(self.encode(context), dtype=torch.long, device=transformer.device)
                context = torch.cat(self._get_batch(data=context))
        # context is (B,T) array of indices
            for _ in range(max_new_tokens):
                # crop context to the last block_size tokens
                context_cond = context[:, -transformer.block_size:]
                logits, loss = self(context_cond) # Does the prediction 
                logits = logits[:, -1, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
                probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, ook (B,C)
                context_next = torch.multinomial(probs, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
                context = torch.cat((context, context_next), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
            return context


        def _get_batch(self, split=None, data=None):
            # Generate a small batch of inputs x and targets y
            data = data if data != None else self.train_data if split =='train' else self.val_data
            ix = torch.randint(len(data) - transformer.block_size, (transformer.batch_size,))
            x = torch.stack([data[i:i+transformer.block_size] for i in ix])
            y = torch.stack([data[i+1:i+transformer.block_size+1] for i in ix])
            x, y = x.to(transformer.device), y.to(transformer.device)
            return x, y


    # Basic helper modules, build an interface when needed
    # Niet zeker of dit werkt? We komen er achter lol
    class Head(nn.Module):
        def __init__(self, transformer, head_size):
            super().__init__()
            self.key = nn.Linear(transformer.n_embed, head_size, bias=False)
            self.query = nn.Linear(transformer.n_embed, head_size, bias=False)
            self.value = nn.Linear(transformer.n_embed, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(transformer.block_size, transformer.block_size)))

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
        def __init__(self, transformer, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([transformer.Head(transformer, head_size) for _ in range(num_heads)])
            self.projection = nn.Linear(transformer.n_embed, transformer.n_embed)
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
        def __init__(self, transformer, n_embed, n_head) -> None:
            super().__init__()
            head_size = n_embed // n_head
            self.sa = transformer.MultiHeadAttention(transformer, n_head, head_size)
            self.ffwd = transformer.FeedFoward(n_embed)
            # This will be a slight deviation from "attention is all you need"
            # We will be doing pre-attention layer normalization
            self.ln1 = nn.LayerNorm(n_embed)
            self.ln2 = nn.LayerNorm(n_embed)
        
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln1(x))
            return x 


if __name__ == "__main__": 
    transformer = Transformer() # Instantiate the transformer
    transformer.optimize() # Optimize (i.e. train) the trainsformer

    print(f"train data {len(transformer.model.train_data)}: {transformer.model.train_data[:128]} and \nval data {len(transformer.model.val_data)}: {transformer.model.val_data[:128]}")

    context = "Wat is uw naam?"
    transformer.model.generate(torch.zeros((1, 1), dtype=torch.long, device=transformer.device)) # Context is, basically, the input string)
    generated = transformer.model.generate(context)[0].tolist()
    print(generated)
    print(transformer.model.decode(generated))
    