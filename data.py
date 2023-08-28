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
from tqdm import tqdm
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
        self.data_dir = data_dir
    
    def setup(self, stage):
        n_samples = len(self.json_data)
        
        n_train_samples = round(n_samples*0.8)
        n_valid_test_samples = n_samples - n_train_samples
        n_valid_samples = round(n_valid_test_samples*0.5)
        n_test_samples = n_valid_test_samples - n_valid_samples
        
        entire_dataset = DocDataset(self.json_data, self.data_dir)
        
        self.train_dataset, valid_test_dataset = random_split(entire_dataset, [n_train_samples, n_valid_test_samples])
        self.valid_dataset, self.test_dataset = random_split(valid_test_dataset, [n_valid_samples, n_test_samples])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )

class DocDataset(Dataset):
    def __init__(self, data, data_dir, load_into_ram = True):
        super().__init__()

        self.data = data
        self.data_dir = data_dir
        self.load_into_ram = load_into_ram
        if self.load_into_ram:
            self.new_data = self.load_data_into_ram(self.data, "Loading data") 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # image = self.load_image(self.data_list[index]['image_path'])
        # mask = self.load_image(self.data_list[index]['mask_path'])
        if self.load_into_ram:
            return self.new_data[index]
        return self.data[index]
    
    def load_data_into_ram(self, data_list, ms):
        temp = []
        print(ms)
        for sample in tqdm(data_list):
            temp.append((self.load_image(sample['image_path']), self.load_image(sample['mask_path'])))
        return temp
    
    def load_image(self, path):
        path = self.data_dir +"/"+ path
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (256,256))
        image = transforms.ToTensor()(image)

        return image