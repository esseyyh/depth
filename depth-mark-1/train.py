import torchvision.transforms as transforms
import torch.nn as nn
#import torchvision
#import math
import matplotlib.pyplot as plt
import torch
#import urllib
import numpy as np
import sys
from src.diff.model import UNet
import PIL
from PIL  import Image 
import hydra

#from torch.util.dataloader import CustomDataset
from src.diff.diffusion import DiffusionModel
from src.diff.AE import Encoder
from src.diff.AE import Decoder

image2=Image.open("image.jpg")



transform = transforms.Compose([
    transforms.Resize((64,64)), # Resize the input image
    transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
])


reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
    transforms.ToPILImage(), # Convert to PIL image
])




@hydra.main(version_base=None,config_path="config",config_name="config")
def train (cfg):
    NO_EPOCHS = cfg.epoch
    PRINT_FREQUENCY = 1
    LR = 0.001
    BATCH_SIZE = 1
    unet = UNet()
    optimizer = torch.optim.Adam(unet.parameters(), lr=LR)
    pil_image = image2
    torch_image = transform(pil_image)

    diffusion_model = DiffusionModel()


    for epoch in range(NO_EPOCHS):
        mean_epoch_loss = []
    
        batch = torch.stack([torch_image] * BATCH_SIZE)
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long()

        batch_noisy, noise = diffusion_model.forward(batch, t) 
        predicted_noise = unet(batch_noisy, t)
    
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(noise, predicted_noise) 
        mean_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
        if epoch % PRINT_FREQUENCY == 0:


       
            sys.stdout.write(" ")
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")


if __name__=="__main__":
    train()
