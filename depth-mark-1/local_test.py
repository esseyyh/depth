import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader,Dataset,Subset

from src.vit import ViT
from utils.data import ImageDataset



@hydra.main(version_base=None,config_path="config",config_name="config")
def train (cfg):
    dataset=ImageDataset("/home/essey/Documents/Ml/datastore/ViT-512/","/home/essey/Documents/Ml/datastore/joined-data-512.csv")#(cfg.data.root_dir,cfg.data.csv_dir)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7,0.3])
    
    train_indices = torch.arange(len(dataset))[:int(cfg.data.train_split * len(dataset))]
    test_indices = torch.arange(len(dataset))[int(cfg.data.train_split * len(dataset)):]

# Create training and test subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

# Create data loaders for the training and test subsets
    train_loader = DataLoader(train_subset,shuffle= cfg.data.train_shuffle, batch_size=2)#,num_workers=4)
    test_loader = DataLoader(test_subset, shuffle=cfg.data.test_shuffle, batch_size=2)#,num_workers=4)

    model = ViT().to("cuda:0")
    #model = torch.nn.DataParallel(model, device_ids=devices)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.LR_1)
    for epoch in range(cfg.params.no_epoch):
        mean_epoch_loss=[]
        for batch in train_loader:



            
    
            batch_image= batch[0]
            batch_image=batch_image.to("cuda:0") 
            images = model(batch_image)
            print(batch_image.shape)
            print(images.shape)
           

            batch_depth= batch[1]
            batch_depth=batch_depth.to("cuda:0") 
            print(batch_depth.shape) 
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(images,batch_depth) 
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            print("test run")
            print(f"loss : {loss}")
            print(images.shape)
    
        if epoch % cfg.params.save_fre == 0:
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")
            torch.save(model,"model.pt")

    print("#######")
    print("#################")
    print("#######")
    #for epoch in range(cfg.params.no_epoch):
    #    mean_epoch_loss=[]
    #    for batch in test_loader:



            
    
     #       batch_image  = batch
     #       batch_image=batch_image.to("cuda:0")
  

     #       images = model(batch_image,train,False)
    
     #       optimizer.zero_grad()
     #       loss = torch.nn.functional.mse_loss(batch_image,images) 
     #       mean_epoch_loss.append(loss.item())
  
     #   if epoch % cfg.params.no_epoch ==0:
     #           print('---')
     #           print (f"Epoch :{epoch}|testloss {np.mean(mean_epoch_loss)} ")
    


if __name__=="__main__":
    train()






