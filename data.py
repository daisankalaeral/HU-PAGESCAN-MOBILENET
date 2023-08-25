import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import lightning as pl
import cv2 as cv
import json

def json_load(path):
    with open(path, "r") as f:
        return json.load(f)

class DocDataModule(pl.LightningDataModule):
    def __init__(self, json_path, data_dir, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.json_data = json_load(json_path)
        print(len(self.json_data))
        self.data_dir = data_dir
    
    def setup(self, stage):
        n_samples = len(self.json_data)
        
        n_train_samples = round(n_samples*0.8)
        n_valid_test_samples = n_samples - n_train_samples
        n_valid_samples = round(n_valid_test_samples*0.8)
        n_test_samples = n_valid_test_samples - n_valid_samples

        self.train_list, valid_test_list = random_split(self.json_data, [n_train_samples, n_valid_test_samples])
        self.valid_list, self.test_list = random_split(valid_test_list, [n_valid_samples, n_test_samples])

    def train_dataloader(self):
        return DataLoader(
            DocDataset(self.train_list, self.data_dir),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            DocDataset(self.valid_list, self.data_dir),
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            DocDataset(self.test_list, self.data_dir),
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )

class DocDataset(Dataset):
    def __init__(self, data_list, data_dir):
        super().__init__()

        self.data_list = data_list
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image = self.load_image(self.data_list[index]['image_path'])
        mask = self.load_image(self.data_list[index]['mask_path'])
        
        return image, mask
        
    def load_image(self, path):
        path = self.data_dir +"/"+ path
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (256,256))
        image = transforms.ToTensor()(image)

        return image
