from torch.utils.data import Dataset
import torch
import os
from torchvision.io import read_image
import random

class ImagesDataset(Dataset):
    def __init__(self, data, targets, data_dir, transform=None):

        self.data_dir = data_dir
        self.target = targets
        self.data = data
        self.transforms = transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = read_image(self.data_dir+self.data[idx])[:-1,:,:]
        img = img.to(torch.float32)
        img = img/255
        target = self.target[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, torch.Tensor(target)