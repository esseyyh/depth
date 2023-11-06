import torch
import torch.nn as nn






class convblock(nn.Module):
        
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
