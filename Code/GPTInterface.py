import torch
import torch.nn as nn
from torch.nn import functional as F

from helpers import get_input
from time import sleep

class Transformer():
    def __init__(self, Model: "Model"=None, m_state=None, **kwargs): # TODO ADD DEBUG WITH PRINTS
        """
        Accepts batch_size=32, block_size=8, max_iters=600
        eval_interval=300, learning_rate=1e-3, device=cuda if available else cpu
        eval_iters=200, embedding_dim=384,
        
        train_data,val_data and tokenizer (all these have fallback options)
        """
        # Passable hyperparameters
        # TODO Implementeer getters voor kwargs
        self.batch_size = kwargs.get("batch_size", 64) # how many independent sequences will we process in parallel?
        self.block_size = 16 # What is the maximum context length for predictions?
        self.max_iters = 12000 # 10k
        self.eval_interval = 300
        self.learning_rate = 1e-3
        self.eval_iters = 200
        self.embedding_dim = 384 # embeddings in dimensions. N = dimensions!
        self.n_head = 6 # Number of heads -> has to neatly divide embedding_dim
        self.n_layer = 6 # Number of blocks
        self.dropout = 0.2 # Dropout randomly drops some blocks during the training, meaning it strongly prevents overfitting

        # Arbitrarily accept all keywords passed
        [setattr(self, key, value) for key, value in kwargs.items()] 

        # Non-passable arguments
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialization of the model TODO What if one wants to load a pretrained model?

        # Dit is slechte code maar ik zit in een college en ben ziek
        # Laat het gepasseerde model, anders laad de state_dict, anders creeer een nieuw model
        if Model != None:
            self.model = Model
        elif m_state != None:
            with torch.device('meta'):
                self.model = self.Model(self)
            self.model.load_state_dict(m_state, assign=True)
        else:
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
        def __init__(self, Transformer: 'Transformer', Tokenizer=None, data=None):
            super().__init__()
            self.Transformer = Transformer
            #-----------------
            
            # Tokenizer
            if Tokenizer != None:
                self.Tokenizer = Tokenizer
                vocab_size = Tokenizer.vocab_size
                chars = Tokenizer.chars
            else:
                print("WARNING: NO TOKENIZER PASSED, USING FALLBACK")
                ### Maak een dataset voor de tokenizer
                self.Tokenizer = None
                text = get_input("wouter")
                chars = sorted(list(set(text)))
                vocab_size = len(chars)
                ### Instantieer het backup encoding algoritme:
                stringtoint = { ch:i for i,ch in enumerate(chars) } # {"A":0, "B":1, ..., "!": 80}
                inttostring= { i:ch for i,ch in enumerate(chars) } # {0:"A", 1:"B", ..., 80:"!"}
                self._bencode = lambda s: [stringtoint[c] for c in s] # backup encoding algo
                self._bdecode = lambda l: ''.join([inttostring[i] for i in l]) # backup decoding algo

            self.chars = chars
            self.vocab_size = vocab_size

            # Data getter
            data = data if data != None else torch.tensor(self.encode(text), dtype=torch.long)
            n = int(0.9*len(data))
            self.train_data = data[:n]
            self.val_data = data[n:]

            # Each token directly reads off the logits for the next token from a lookup table (which lookup table?)
            self.tokeembedding_dimding_table = nn.Embedding(self.vocab_size, self.Transformer.embedding_dim)

            """Note that the sequence they appear is also the sequence they are used"""

            #We're not just encoding identity, we're also encoding position!
            self.positioembedding_dimding_table = nn.Embedding(self.Transformer.block_size, self.Transformer.embedding_dim)
            self.blocks = nn.Sequential(*[self.Transformer.Block(self.Transformer, self.Transformer.embedding_dim, n_head=self.Transformer.n_head) for _ in range(self.Transformer.n_layer)])
            self.ln_f = nn.LayerNorm(self.Transformer.embedding_dim) # Final layer norm
            self.lm_head = nn.Linear(self.Transformer.embedding_dim, self.vocab_size) # LM=loaded model
            # N_embed is the number of embedded dimentions
            # .Embedding creates a shape of vocab_size x vocab_size
            # De inputs voor de transformer zoeken in de tensor rij en plukken de Xte (X=tokenized input integer) rij uit de lookup table
            
        def encode(self, string):
            """
            Calls tokenizer.encode, else falls to backup
            """
            if self.Tokenizer != None:
                self.Tokenizer.encode(string)
            else:
                return self._bencode(string)

        def decode(self, string):
            """ 
            Calls tokenizer.decode, else falls to backup
            """
            if self.Tokenizer != None:
                self.Tokenizer.decode(string)
            else:
                return self._bdecode(string)       

        def forward(self, context, targets=None):
            B, T = context.shape
            #context and targets are both (B,T) tensor of integers
            tok_em = self.tokeembedding_dimding_table(context)  # B,T,C
            pos_em = self.positioembedding_dimding_table(torch.arange(T, device=self.Transformer.device)) # T, C
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
        
        def generate(self, context, max_new_tokens=32):
            """
            Requires a tensor as context input (encoded using the same tokenizer),
            max_new_tokens defaults to 512

            returns encoded tensor response
            """ # TODO Just generating the answer (no training) is already very slow on NR2, I legit need cuda
            print(f"Textuele context: {context}")
            if type(context) == str:
                while len(context) <= self.Transformer.block_size: # Terwijl onder minimale lengte voor batching
                    context = context.ljust(len(context)  + 1) # Voeg één spatie 
                context = torch.tensor(self.encode(context), dtype=torch.long, device=self.Transformer.device)
                context = torch.cat(self._get_batch(data=context))
            print(f"Tensorized context: {len(context)} ; {context}")
        # context is (B,T) array of indices
            for _ in range(max_new_tokens):
                print(f"Currently generating token {_}")
                # crop context to the last block_size tokens
                context_cond = context[:, -self.Transformer.block_size:]
                logits, loss = self(context_cond) # Does the prediction 
                logits = logits[:, -1, :] # Foxus only the last time step, (B,C), de -1 skipt de T dimensie
                probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, ook (B,C)
                context_next = torch.multinomial(probs, num_samples=1) # Sample from the distributino by flattening it, (B, 1)
                context = torch.cat((context, context_next), dim=1) # ( Append sampled index to the running sequence) (B, T+1)
            return context

        def _get_batch(self, split=None, data=None):
            # Generate a small batch of inputs x and targets y
            data = data if data != None else self.train_data if split =='train' else self.val_data
            ix = torch.randint(len(data) - self.Transformer.block_size, (self.Transformer.batch_size,))
            x = torch.stack([data[i:i+self.Transformer.block_size] for i in ix])
            y = torch.stack([data[i+1:i+self.Transformer.block_size+1] for i in ix])
            x, y = x.to(self.Transformer.device), y.to(self.Transformer.device)
            return x, y

    def save(self, path="Code/models/default.pth"): # PTH is convention according to the docs
        torch.save(self.model, path)
        return
    
    def save_std(self, filename="default"): # PT is convention according to the docs
        path = "Code/models/" + filename + "pth"
        torch.save(self.model.state_dict(), path)
        return

    # Basic helper modules, build an interface when needed
    # Niet zeker of dit werkt? We komen er achter lol
    class Head(nn.Module):
        def __init__(self, Transformer: 'Transformer', head_size):
            super().__init__()
            self.Transformer = Transformer
            self.key = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.query = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.value = nn.Linear(self.Transformer.embedding_dim, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(self.Transformer.block_size, self.Transformer.block_size)))
            self.dropout = nn.Dropout(self.Transformer.dropout)
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
        def __init__(self,  Transformer: 'Transformer', num_heads, head_size):
            super().__init__()
            self.Transformer = Transformer
            self.heads = nn.ModuleList([self.Transformer.Head(self.Transformer, head_size) for _ in range(num_heads)])
            self.projection = nn.Linear(self.Transformer.embedding_dim, self.Transformer.embedding_dim) # patch embedding v encoder embedding
            self.dropout = nn.Dropout(self.Transformer.dropout)
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.projection(out)) # A linear transformation of the output of the concationation
            return out
    class FeedFoward(nn.Module):
        """A simple linear layer followed by a non-linearity"""
        def __init__(self,  Transformer: 'Transformer', embedding_dim):
            super().__init__()
            self.Transformer = Transformer
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim),
                nn.Dropout(self.Transformer.dropout)
            )
        def forward(self, x):
            return self.net(x)
    class Block(nn.Module):
        def __init__(self, Transformer: 'Transformer', embedding_dim, n_head) -> None:
            super().__init__()
            self.Transformer = Transformer
            head_size = embedding_dim // n_head
            self.sa = self.Transformer.MultiHeadAttention(self.Transformer, n_head, head_size)
            self.ffwd = self.Transformer.FeedFoward(self.Transformer, embedding_dim)
            # This will be a slight deviation from "attention is all you need"
            # We will be doing pre-attention layer normalization
            self.ln1 = nn.LayerNorm(embedding_dim)
            self.ln2 = nn.LayerNorm(embedding_dim)
        
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x 


if __name__ == "__main__": 
    transformer = Transformer() # Instantiate the 
    # print("Instantiated Transformer")
    transformer.optimize() # Optimize (i.e. train) the trainsformer
    # print("Optimized Transformer")
    # # print(f"train data {len(transformer.model.train_data)}: {transformer.model.train_data[:128]} and \nval data {len(transformer.model.val_data)}: {transformer.model.val_data[:128]}")

    # context = str(input("Hey :) Please talk to me! \n"))
    # sleep(.15)
    # generated = transformer.model.generate(context)[0].tolist()
    # print(f"{type(generated)} response: {generated}")
    # decoded = transformer.model.decode(generated)
    # print(f"decoded: {decoded}")
    transformer.save_std("v4")
    print('saved transformer')

    
