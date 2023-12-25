
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
    