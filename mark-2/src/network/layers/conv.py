import torch.nn as nn
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