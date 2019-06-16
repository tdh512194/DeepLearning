import numpy as np
import pandas as pd
import os
import scipy.io
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import gc
from Unet import Unet
import torch
import sys
import glob

use_gpu = torch.cuda.is_available()

# create a vectorizer function returning 0 if x = 0 and 1 otherwise
thresh = np.vectorize(lambda x: 0 if x == 0 else 1, otypes=[np.float])

def create_dataset(paths, width_in, height_in, width_out, height_out, data_indexes):
    x = []
    y = []
    for path in tqdm(paths):
        # load dataset, mat is a dict
        mat = scipy.io.loadmat(path)
        img_tensor = mat['images']
        fluid_tensor = mat['manualFluid1']

        # create input x
        img_array = np.transpose(img_tensor, (2, 0, 1)) / 255 # channel first, scale to 0..1
        img_array = resize(img_array, (img_array.shape[0], width_in, height_in)) # resize
        
        # create label y
        fluid_array = np.transpose(fluid_tensor, (2, 0, 1)) # channel first
        fluid_array = thresh(fluid_array) # scale to 0 or 1
        fluid_array = resize(fluid_array, (fluid_array.shape[0], width_out, height_out))

        # add to x, y
        for idx in data_indexes:
            x += [np.expand_dims(img_array[idx], 0)]
            y += [np.expand_dims(fluid_array[idx], 0)]
    return np.array(x), np.array(y)

def get_dataset(width_in, height_in, width_out, height_out):
    input_path = '2015_BOE_Chiu'
    subject_path = os.listdir(input_path)
    m = len(subject_path)
    data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]

    x_train, y_train = create_dataset(subject_path[:-1], width_in, height_in, width_out, height_out, data_indexes)
    x_val, y_val = create_dataset(subject_path[-1:], width_in, height_in, width_out, height_out, data_indexes)
    return x_train, y_train, x_val, y_val

def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()

    # forward
    outputs = unet(inputs)                # (batch size, n classes, img cols, img rows)
    outputs = outputs.permute(0, 2, 3, 1) # (batch size, img cols, img rows, n classes)
    m = outputs.shape[0]
    
    # calculate loss
    outputs = outputs.resize(m*width_out*height_out, 2) # 2 is n classes
    labels = labels.resize(m*width_out*height_out)
    loss = criterion(outputs, labels)

    # calculate gradient
    loss.backward()
    # update
    optimizer.step()
    return loss

def get_val_loss(x_val, y_val, width_out, height_out, unet):
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
    if use_gpu:
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    m = x_val.shape[0]
    outputs = unet(x_val)
    outputs = outputs.permute(0, 2, 3, 1)
    
    outputs = outputs.resize(m*width_out*height_out, 2) # 2 is n classes
    labels = y_val.resize(m*width_out*height_out)

    loss = F.cross_entropy(outputs, labels)
    return loss.data







    
        
