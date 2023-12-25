import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from src.network.encoder import Block
from src.network.layers.conv import convblock

class ViT(nn.Module):

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
