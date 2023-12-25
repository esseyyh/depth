
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from src.network.patch_embedding import PatchEmbedding
from src.network.layers.feedfw import FeedFoward
from src.network.multihead import MultiHeadAttention

class Block(nn.Module):
    """ Transformer and patching block """

    def __init__(self,size,dims, n_head=12,patch_size=12,dropout=0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.im_size=size
        self.n_embd=patch_size*patch_size*dims
        self.patches=(self.im_size/patch_size)**2
        self.head_size = self.n_embd // n_head
        self.patch= PatchEmbedding (self.im_size,patch_size,dims,self.n_embd)
        self.sa = MultiHeadAttention(n_head,self.head_size,self.n_embd,self.patches)
        self.ffwd = FeedFoward(self.n_embd)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        
        x=self.patch(x)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        
        B,T,C=x.shape
        x=torch.reshape(x,(B,self.im_size,self.im_size,-1))
        x=x.permute(0,3,1,2)
        return x