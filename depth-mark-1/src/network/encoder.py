import torch
import torch.nn as nn

from src.network.multihead import MultiHeadAttention
from src.network.layers.ffd import FeedFoward
from src.network.patching import PatchEmbedding



class Block(nn.Module):
    """ Transformer and patching block """

    def __init__(self, n_embd, n_head,dropout=0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.n_embd=n_embd
        self.head_size = n_embd // n_head
        self.patch=PatchEmbedding()
        self.sa = MultiHeadAttention(n_head,self.head_size,n_embd)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        
        x=self.patch(x)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        
        
        B,T,C=x.shape
        x=torch.reshape(x,(B,32,32,self.n_embd))
        x=x.permute(0,3,1,2)
        return x
    