import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np




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
        x=torch.reshape(x,(B,36,36,self.n_embd))
        x=x.permute(0,3,1,2)
        return x
    

class decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=Block(768,12)
            
            

            
            self.channels=[768,512,256,128,64]
            self.block = nn.Sequential(*[blocks(ch1, ch2) for ch1, ch2 in zip(self.channels, self.channels[1:])])
            self.convf=nn.Conv2d(self.channels[-1],1,3,padding=1)
          
        def forward(self,x):
            x=self.encoder(x)
            x=self.block(x)
            x=self.convf(x)
            
            return x
        