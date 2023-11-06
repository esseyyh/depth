import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader

from src.vit import depth_model
from utils.data import ImageDataset



@hydra.main(version_base=None,config_path="config",config_name="config")
def train (cfg):
    dataset=ImageDataset("/home/essey/Documents/Ml/datastore/ViT-512","/home/essey/Documents/Ml/scripts/image_extraction/mesh-vis/src/joined-data-512.csv")
    
    NO_EPOCHS = cfg.params.no_epoch
    PRINT_FREQUENCY = cfg.params.print_fre
    LR = cfg.params.LR

    model = depth_model(cfg.hparams1).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data_loader = DataLoader(dataset,cfg.params.batch_size)
    print("train loss for model")

    for epoch in range(NO_EPOCHS):
        mean_epoch_loss=[]
        for batch in data_loader:



            
    
            batch_image , batch_depth = batch
            batch_depth=batch_depth.to("cuda")
            batch_image=batch_image.to("cuda")
  

            predicted_depth = model(batch_image)
    
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(batch_depth, predicted_depth) 
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
    
        if epoch % PRINT_FREQUENCY == 0:
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")
            torch.save(model,"out/model.pt")





if __name__=="__main__":
    train()


