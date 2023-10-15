import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np



class PatchEmbedding(nn.Module):

    """splts the input image to multiple patches and then embed them in a single flattned vector to be fed to transformer """
    

    def __init__(self,image_size=576,patch_size=16,in_channels=3,embed_dim=768):
        super().__init__()
        
        self.image_size=image_size
        self.patch_size=patch_size
        self.num_patches=(image_size/patch_size)**2
        self.proj= nn.Conv2d( in_channels, embed_dim, kernel_size=patch_size,  stride=patch_size )


    def forward(self,x):
        x=self.proj(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        return x
    


class SingleHeadAttention(nn.Module):
    def __init__(self,head_size,n_embed=768,drop_out=0,patches=1296):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.dropout=nn.Dropout(drop_out)
        self.register_buffer('tril',torch.tril(torch.ones(patches,patches)))
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
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,n_embd,dropout=0):
        super().__init__()
        
       
        self.heads = nn.ModuleList([SingleHeadAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd,dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

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


class blocks(nn.Module):
        
        def __init__(self,ch1,ch2):
            super().__init__()
        
            self.up=nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv1=nn.Conv2d(ch1,ch2,3,padding=1)
            self.conv2=nn.Conv2d(ch2,ch2,3,padding=1)
            
        def forward(self,x):
            
            x=self.up(x)
            x=self.conv1(x)
            x=self.conv2(x)
            
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
        
