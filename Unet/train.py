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
from torch.nn import functional as F
import sys
import glob

# use_gpu = torch.cuda.is_available()
use_gpu = False

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
    input_path = '/home/ted/Projects/DeepLearning/Unet/2015_BOE_Chiu'
    subject_path = glob.glob(input_path + '/*.mat')
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
    y_val = torch.from_numpy(y_val).long()
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

def train(unet, batch_size, epochs, epoch_lapse, threshold,
          learning_rate, criterion, optimizer,
          x_train, y_train, x_val, y_val, 
          width_out, height_out):
    epoch_iter = x_train.shape[0] // batch_size
    t = trange(epochs, leave=True)
    for epoch in t:
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.from_numpy(x_train[i * batch_size:(i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(y_train[i * batch_size:(i + 1) * batch_size]).long()
            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(inputs=batch_train_x, labels=batch_train_y, 
                                    optimizer=optimizer, criterion=criterion, 
                                    unet=unet, width_out=width_out, height_out=height_out)
            total_loss += batch_loss
        if (epoch + 1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out, unet)
            print("Epoch {:02d}: _Train loss: {:.4f} _ Val loss: {:.4f}".format(epoch + 1, total_loss, val_loss))
            torch.save(unet.state_dict, 'unet.pt')
        
def plot_examples(unet, datax, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_idx = np.random.randint(m)
        image_arr = unet(torch.from_numpy(datax[image_idx:image_idx + 1]).float()).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_idx], (1,2,0))[:,:,0])
        ax[row_num][1].imshow(np.transpose(image_arr, (1,2,0))[:,:,0])
        ax[row_num][2].imshow(image_arr.argmax(0))
        ax[row_num][3].imshow(np.transpose(datay[image_idx], (1,2,0))[:,:,0])
    plt.show()

def main():
    width_in = 284
    height_in = 284
    width_out = 116
    height_out = 116
    PATH = './unet.pt'
    x_train, y_train, x_val, y_val = get_dataset(width_in, height_in, width_out, height_out)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    batch_size = 2
    epochs = 5
    epoch_lapse = 2
    threshold = 0.5
    learning_rate = 0.01
    unet = Unet(in_channels=1,out_channels=2)
    if use_gpu:
        unet = unet.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
    if True:
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate, criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
    else:
        if use_gpu:
            unet.load_state_dict(torch.load(PATH))
        else:
            unet.load_state_dict(torch.load(PATH, map_location='cpu'))
        print(unet.eval())
    plot_examples(unet, x_train, y_train)
    plot_examples(unet, x_val, y_val)

if __name__ == "__main__":
    main()
    
    
        
