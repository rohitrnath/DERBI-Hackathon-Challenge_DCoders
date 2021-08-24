import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image

class NIH_Dataset(Dataset):

    def __init__(self, data, img_dir, transform=None, aug=None):
        self.data = data
        self.img_dir = img_dir 
        self.transform = transform 
        self.aug = aug

    def __len__(self):
        return len(self.data)
    
    def __in__(self, idx):
        return self.img_dir + self.data.iloc[:,0][idx]

    def __getitem__(self, idx):
        img_file = self.img_dir + self.data.iloc[:,0][idx]
        img = Image.open(img_file).convert('RGB')
        label = np.array(self.data.iloc[:,1:].iloc[idx])
        if self.aug:
            data = {"image": np.array(img)}
            img = self.aug(**data)['image']
            #img = np.transpose(img, (2, 0, 1))
        elif self.transform:
            img = self.transform(img)

        return img,label