from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.imgs = []
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".png"):
                    self.imgs.append(os.path.join(root, file))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        # img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
    
    # sourced from https://github.com/wherobots/GeoTorchAI/blob/main/geotorchai/datasets/grid/processed.py
    def set_sequential_representation(self, history_length, predict_length):
        '''
        Call this method if you want to iterate the dataset as a sequence of histories and predictions instead of closeness, period, and trend.

        Parameters
        ..........
        history_length (Int) - Length of history data in sequence of each sample
        predict_length (Int) - Length of prediction data in sequence of each sample
        '''

        history_data = []
        predict_data = []
        seq_length = history_length + predict_length
        total_length = len(self.imgs)
        for end_idx in range(history_length + predict_length, total_length, seq_length):
            history_idx = list(range(end_idx-predict_length-history_length, 
                                     end_idx-predict_length))
            predict_idx = list(range(end_idx-predict_length, 
                                     end_idx))

            history_frames = [np.array(self.__getitem__(idx)) for idx in history_idx]
            predict_frames = [np.array(self.__getitem__(idx)) for idx in predict_idx]
            
            history_data.append(history_frames)
            predict_data.append(predict_frames)

        history_data = np.stack(history_data)
        predict_data = np.stack(predict_data)

        self.X_data = torch.tensor(history_data)
        self.Y_data = torch.tensor(predict_data)

class ImageDataset(ImageFolder):
    def __init__(self, ds):
        self.X_data = ds.X_data
        self.Y_data = ds.Y_data

    def __getitem__(self, index):
        x_data = self.X_data[index]
        y_data = self.Y_data[index]
        
        # Returning a dictionary containing data elements
        return {"X_data": x_data, "Y_data": y_data}

    def __len__(self):
        return len(self.X_data)