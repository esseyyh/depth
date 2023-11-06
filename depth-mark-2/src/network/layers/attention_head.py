import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class SingleHeadAttention(nn.Module):
    def __init__(self,head_size,n_embed,patches,drop_out=0):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.dropout=nn.Dropout(drop_out)
        self.patches=int(patches)
        self.register_buffer('tril',torch.tril(torch.ones(self.patches,self.patches)))
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        x=torch.matmul(q,k.transpose(-2,-1)) * C**-0.5
        x=x.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        x=F.softmax(x,dim=-1)
      
        x=torch.matmul(x,v)
       
        return x
    
