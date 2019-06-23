import os
import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from PIL import Image

class SegmentationDataset(Dataset):
    """
    Semantic Segmentation dataset
    X: image 
    Y: masked image
    Yields:
        {
            'input': np.array of shape (h, w, c),
            'label': np.array of shape (h, w, 1)
        }
    """

    def __init__(self, csv_path, root_input_dir, root_label_dir, 
                 input_ext, label_ext, label_prefix='', label_suffix='', transform=None):
        """
        Args:
            csv_path (string):       path to csv file containing names of imgs
            root_input_dir (string): path to the folder containing the input imgs
            root_label_dir (string): path to the folder containing the label imgs
        """
        self.img_names = pd.read_csv(csv_path)
        self.root_input_dir = root_input_dir
        self.root_label_dir = root_label_dir
        self.label_prefix = label_prefix
        self.label_suffix = label_suffix
        self.input_ext = input_ext
        self.label_ext = label_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        base_img_name = str(self.img_names.iloc[idx, 0])
        img_name = os.path.join(self.root_input_dir, base_img_name + '.' + self.input_ext)
        input_img = cv2.imread(img_name)
        input_img = self.format_input(input_img)
        label_name = os.path.join(self.root_label_dir, self.label_prefix + base_img_name + self.label_suffix + '.' + self.label_ext)
        label_img = cv2.imread(label_name)
        label_img = self.format_label(label_img)

        if self.transform:
            input_img = self.transform['input'](input_img)
            label_img = self.transform['label'](label_img)
        
        input_img = input_img.float()
        # convert PIL -> nparray -> torch tensor (to preserve value range we dont use ToTensor() directly)
        label_img = np.array(label_img)
        label_img = torch.from_numpy(label_img).long()

        sample = {'input': input_img, 'label': label_img}
        """
        shape output using dataloader with this dataset:
        {
            'input': (m, c, h, w), 'label': (m, c, h, w)
        }
        """
        return sample

    def format_input(self, x):
        x = Image.fromarray(x)
        return x

    def format_label(self, x):
        # tranpose to channel first
        x = np.transpose(x, (2, 0, 1)) # (c, h, w)
        # binarize the label image
        x = np.vectorize(lambda x: 0 if x == 255 else 1)(x)
        # convert to 1 channels from 3 channels
        x = np.sum(x, axis=0)
        x = np.vectorize(lambda x: 1 if x > 0 else 0)(x)
        x = Image.fromarray(x, 'P')
        return x
