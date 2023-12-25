import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np





class PatchEmbedding(nn.Module):

    """splts the input image to multiple patches and then embed them in a single flattned vector to be fed to transformer """
    

    def __init__(self,image_size,patch_size,in_channels,embed_dim):
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

    def __init__(self,size,dims, n_head=12,patch_size=12,dropout=0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.im_size=size
        self.n_embd=patch_size*patch_size*dims
        self.patches=(self.im_size/patch_size)**2
        self.head_size = self.n_embd // n_head
        self.patch=PatchEmbedding(self.im_size,patch_size,dims,self.n_embd)
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

class convblock(nn.Module):
    def __init__(self, channels_in, channels_out, downsample=True):
        super().__init__()
        
        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out,3, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, 3, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
            
        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)
        
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bnorm1(self.relu(self.conv1(x)))
        x = self.bnorm2(self.relu(self.conv2(x)))

        return self.final(x)


class depth_model(nn.Module):

    def __init__(self,params):
        super().__init__()

        self.image_sequence=params.size

        self.patch_size=params.patch_size

        self.dims_sequence=params.dims
        
        self.in_im_size=params.initial_image_size
        
        
        
        self.n_heads=params.n_heads
        

        self.first_block=Block(576,3,self.n_heads,self.patch_size)


        self.block=nn.ModuleList([Block(size,dims,self.n_heads,self.patch_size)for size,dims in zip(self.image_sequence,self.dims_sequence[1:])])
        


        self.downsampling = nn.ModuleList([convblock(channels_in,
                                                 channels_out,
                                                 ) for channels_in, channels_out in zip(self.dims_sequence, self.dims_sequence[1:])])
         
        self.upsampling = nn.ModuleList([convblock(channels_in,
                                               channels_out,
                                               downsample=False) for channels_in, channels_out in zip(self.dims_sequence[::-1], self.dims_sequence[::-1][1:])])
        
        self.final=nn.Conv2d(3,1,3,1,1)




    def forward(self, x):
        residuals=[]

        

        for i in range(0, len(self.block)):
            x=self.downsampling[i](x)

            x = self.block[i](x)
            residuals.append(x)

        for us, res in zip(self.upsampling, reversed(residuals)):
            # concatinaing and upsampling 
            x = us(torch.cat((x, res), dim=1))      




        return self.final(x)


