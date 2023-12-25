
import torch.nn as nn


from src.network.layers.conv import convblock
from src.network.encoder import Block

class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=Block(768,12)
            
            

            
            self.channels=[768,512,256,128,64]
            self.decoder = nn.Sequential(*[convblock(ch1, ch2) for ch1, ch2 in zip(self.channels, self.channels[1:])])
            self.convf=nn.Conv2d(self.channels[-1],1,3,padding=1)
          
        def forward(self,x):
            x=self.encoder(x)
            x=self.decoder(x)
            x=self.convf(x)
            
            return x
        
