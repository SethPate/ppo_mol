"""
Transformer code. Owes a great debt to Andrej Karpathy's nanoGPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

def make(cfg):
    """
    Create a fresh Transformer, load the state dict if available.
    """
    model = Transformer(cfg)
    if os.path.exists(cfg['model_f']):
        model.load_state_dict(torch.load(cfg['model_f']))
        print(f"Loaded model from {cfg['model_f']}")
    else:
        print(f"Created new model")
    return model

class Transformer(nn.Module):
    """
    A decoder-only (GPT-style) transformer, comprising layered blocks.
    A block is a self attention layer followed by a feedforward layer.
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg['embed_dim']
        self.n_heads = cfg['n_heads']
        self.n_layers = cfg['n_layers']
        self.n_vocab = cfg['n_vocab']
        self.block_size = cfg['block_size']

        self.embed = nn.Embedding(self.n_vocab, self.embed_dim)
        self.pos = nn.Embedding(self.block_size, self.embed_dim)
        self.drop = nn.Dropout(cfg['dropout'])
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(Block(self.block_size, self.embed_dim,
                self.n_heads, cfg['dropout']))
        self.lm_head = nn.Linear(self.embed_dim, self.n_vocab)

    def forward(self, x):
        """
        Embed x using positional embeddings,
        then run through the transformer's layers.
        Finally, encode x with the language model head.

        inputs: token indices, a tensor of shape (batch, seq, 1)
        returns: logits, a tensor of shape (batch, seq, vocab)
        """
        assert x.ndim == 2, \
            f"input should be (batch, seq), got {x.shape}"
        assert x.shape[1] <= self.block_size, \
            f"input sequence too long, max is {self.block_size}"

        x = self.embed(x) # batch, seq, embed
        pos = torch.arange(0, x.shape[1], \
            device=x.device, dtype=torch.long)
        pos_emb = self.pos(pos) # batch, seq, embed
        x = self.drop(x + pos_emb) # batch, seq, embed
        for i in range(self.n_layers): # self attn
            x = self.layers[i](x) # batch, seq, embed
        x = self.lm_head(x) # batch, seq, vocab

        return x

    @torch.no_grad()
    def sample(self, n, max_new, start_token=0, end_token=1, temp=1.0):
        """
        Generate (n) random samples to judge training.

        For generating molecules in our application, pass
        the correct start token for <START> (default 0).
        Generation will stop when the model predicts the
        end token for <END> (default 1), or hits the limit of
        (max_new) tokens added to the start token.

        Adjust the temperature to flatten or expand the distribution
        over indices at each time step.
        """
        # generate one sample at a token, one token at a time
        generated = []
        for _ in range(n):
            x = torch.tensor([[start_token]], dtype=torch.long)
            for _ in range(max_new):
                # crop sequence to block size
                x_cond = x if x.shape[1] <= self.block_size else \
                    x[:,-self.block_size:]
                # take the logits for only the next token
                logits = self.forward(x_cond)[:,-1,:] # 1, vocab
                # sample from the distribution
                logits = logits / temp
                probs = F.softmax(logits, dim=-1) # 1, vocab
                next_token = torch.multinomial(probs, num_samples=1) # 1, 1
                # add the new token to the sequence
                x = torch.cat((x, next_token), dim=1) # 1, seq
                # if we've reached the end token, stop
                if next_token == end_token:
                    break
            generated.append(x)
        return generated

class Block(nn.Module):
    """
    A transformer block with self-attention, layer norm, and a
    feedforward layer.
    """
    def __init__(self, block_size, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn = SelfAttn(block_size, embed_dim, n_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        With a 'skip' connection; the layers are additive.
        """
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.ff(x))
        return x

class FeedForward(nn.Module):
    # projects to 4 * embed_dim; seems like enough

    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttn(nn.Module):
    """
    Multi-head scaled dot product attention, with layer norm.
    """
    def __init__(self, block_size, embed_dim, n_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.nh = n_heads
        self.attn = nn.Linear(embed_dim, embed_dim * 3) # K, Q, V
        self.ff = FeedForward(embed_dim, dropout)
        self.drop = nn.Dropout(dropout)
        # save causal mask as a buffer to keep it separate from params
        mask = torch.tril(torch.ones(block_size, block_size))
        mask = mask.view(1, 1, block_size, block_size)
        self.register_buffer('mask', mask)

    def forward(self, x):
        """
        inputs: a tensor of shape (batch, seq, embed)
        returns: a tensor of shape (batch, seq, embed)
        """
        assert x.ndim == 3, \
            f"input should be (batch, seq, embed), got {x.shape}"
        B, T, C = x.shape # batch, sequence length, embed dim
        # all three k,q,v at once
        K,Q,V = self.attn(x).split(self.embed_dim, dim=2)

        # break into heads, B, nh, T, hs (C // nh)
        K = K.reshape(B, T, self.nh, C // self.nh).transpose(1,2)
        Q = Q.reshape(B, T, self.nh, C // self.nh).transpose(1,2)
        V = V.reshape(B, T, self.nh, C // self.nh).transpose(1,2)

        # (B nh T hs) @ (B nh hs T) -> (B nh T T)
        att = Q @ K.transpose(-2,-1) 
        att = att / (1.0 * math.sqrt(C // self.nh)) # scaled dot product
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) # B nh T T
        # mask 'future' tokens to prevent cheating
        y = att @ V # (B nh T T) @ (B nh T hs) -> (B nh T hs)
        # put the heads back together for (B, T, C)
        # not sure that contiguous() is necessary
        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.drop(self.ff(y))

        return y
