import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class BuildDataset(Dataset):
    def __init__(self, df, params_loaded):
        self.device = torch.device(params_loaded.device)
        self.params_loaded = params_loaded
        self.preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.df = df

    def __getitem__(self,index):
        result = self.df.iloc[index]
        rating = result['response']
        img_path = self.params_loaded['data']['imgs_path'] + '/' + result['stimulus'][36:44]
        img = Image.open(img_path)
        img = self.preprocess(img)
        
        return img, rating
    
    def __len__(self):
        return self.df.shape[0]


class BuildDataloader(DataLoader):
    def __init__(self,dataset,batch_size,shuffle,num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def collect_fn(self, batch):
        
        out_batch = {}
        img_list = []
        rating_list = []
        
        for img, rating in batch:
          img_list.append(img)
          rating_list.append(rating)  

        out_batch['images'] = torch.stack(img_list,dim=0)
        out_batch['ratings'] = rating_list

        return out_batch
    
    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
