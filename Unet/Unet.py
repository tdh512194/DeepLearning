import os
import json
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
from utils import transform_preprocess
from PIL import Image

class Unet():
    '''
    U-net class
    '''

    def __init__(self, config_path, weights_path=None, is_training=False):
        with open(config_path) as config_file:
            self.config = json.load(config_file)
        self.config['config_path'] = config_path
        # initialize Unet model
        self.model = UnetModel(in_channels=self.config['in_channels'],
                               out_channels=self.config['out_channels'])
        print('Unet model initialized.')
        # define preprocess methods
        self.transform_preprocess = transform_preprocess(config['height_in'], config['width_in'])
        # load weights
        self.weights_path = weights_path
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path))
            print('weights {} loaded.'.format(os.path.basename(self.weights_path)))
        else:
            print('No weights added.')
        # set train or eval mode for layers such as Dropout and Batchnorm
        self.is_training = is_training
        if self.is_training:
            self.model.train()
            print('TRAIN mode.')
        else:
            self.model.eval()
            print('EVAL mode.')

    def process(self, x):
        """
        inference only, x can be either path to image or numpy array
        """
        if isinstance(x, str):
            x = Image.open(x)
        elif isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        # set eval mode
        self.model.eval()

        x = self.preprocess(x)
        output = self.model(x)
        output = self.postprocess(x)

        # set back to train mode to model if neccessary
        if self.is_training:
            self.model.train()
        return output
        
    def preprocess(self, x):
        # x = np.transpose(x, (2, 0, 1))
        # x = x / 255
        return self.transform_preprocess(x)
    
    def postprocess(self, x, resize=False):
        x = np.argmax(x, axis=0) # axis 0 is channel axis
        x = np.transpose(x, (1, 2, 0)) # (h, w, c)
        if resize:
            x = cv2.resize(x, (self.config['height_in'], self.config['width_in']))
        return x

class UnetModel(nn.Module):
    '''
    U-net architecture
    '''

    def __init__(self, in_channels=3, out_channels=2):
        super(Unet, self).__init__()
        # Encode layers
        self.conv_encode1 = self.down_sampling_block(in_channels, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.down_sampling_block(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.down_sampling_block(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.down_sampling_block(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Bottle neck layer
        self.botte_neck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, 
                               stride=2, output_padding=1)
        )

        # Decode layers
        self.conv_decode4 = self.up_sampling_block(1024, 256)
        self.conv_decode3 = self.up_sampling_block(512, 128)
        self.conv_decode2 = self.up_sampling_block(256, 64)

        # Output layer
        self.output = self.output_block(128, out_channels)

    def forward(self, x):
        # Encode layer
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.maxpool4(encode_block4)
        # Bottle neck
        botte_neck1 = self.botte_neck(encode_pool4)
        # Decode layer
        concat_block4 = self.crop_and_concat(botte_neck1, encode_block4, crop=True)
        decode_block4 = self.conv_decode4(concat_block4)
        concat_block3 = self.crop_and_concat(decode_block4, encode_block3, crop=True)
        decode_block3 = self.conv_decode3(concat_block3)
        concat_block2 = self.crop_and_concat(decode_block3, encode_block2, crop=True)
        decode_block2 = self.conv_decode2(concat_block2)
        concat_block1 = self.crop_and_concat(decode_block2, encode_block1, crop=True)
        # Output layer
        output = self.output(concat_block1)
        return output

    def down_sampling_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def up_sampling_block(self, in_channels, out_channels, kernel_size=3):
        mid_channels = in_channels // 2

        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, 
                               stride=2, padding=1, output_padding=1)  
        )
        return block
    
    def output_block(self, in_channels, out_channels, kernel_size=3):
        mid_channels = in_channels // 2

        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, padding=1, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block
    
    def crop_and_concat(self, upsampled, crop_from, crop=False):
        """
        calculate the excessive size -> crop the encode block -> concat with the decode block
        """
        if crop:
            # calculate excessive size of each side (/2)
            crop_size = (crop_from.size()[2] - upsampled.size()[2]) // 2
            # crop to the 2 last dimentions (left, right, top, bottom) as in torch.nn.F.pad()
            crop_from = F.pad(crop_from, (-crop_size, -crop_size, -crop_size, -crop_size))
        block = torch.cat((crop_from, upsampled), dim=1)
        return block
