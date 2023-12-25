
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from src.network.layers.attention_head import SingleHeadAttention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,n_embd,patches,dropout=0):
        super().__init__()
        
       
        self.heads = nn.ModuleList([SingleHeadAttention(head_size,n_embd,patches) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out