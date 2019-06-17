import torch
from torch import nn
import torch.nn.functional as F


class Unet(nn.Module):
    '''
    U-net architecture
    '''

    def __init__(self, in_channels, out_channels):
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
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, output_padding=1)
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
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)  
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