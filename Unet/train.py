from Unet import Unet
from torch.nn.functional import cross_entropy
from utils import transform_preprocess, segmentation_correct, weighted_dice
from dataloader import SegmentationDataset
import json
import copy
import os
import time
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)


def criterion(outputs, labels, weight=None):
    '''
    combine dice loss and cross entropy loss
    '''
    wd_loss = weighted_dice(outputs, labels, weight)
    ce_loss = cross_entropy(outputs, labels, weight)
    loss = 0.4 * ce_loss + 0.6 * wd_loss
    return loss


def train(model, criterion, optimizer, scheduler, n_epochs, batch_size, save_interval, config_path):
    start_train_time = time.time()
    # define training configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_path, 'r') as f:
        config = json.load(f)

    csv_path_train = '/Users/tdh512194/Desktop/Github DL/data/names_test.csv'
    csv_path_val = '/Users/tdh512194/Desktop/Github DL/data/names_test.csv'
    root_input_dir = '/Users/tdh512194/Desktop/Github DL/data/Challenge2_Training_Task12_Images'
    input_ext = 'jpg'
    root_label_dir = '/Users/tdh512194/Desktop/Github DL/data/Challenge2_Training_Task2_GT'
    label_ext = 'bmp'
    label_prefix = ''
    label_suffix = '_GT'

    root_weights_dir = '/Users/tdh512194/Desktop/Github DL/DeepLearning/Unet/weights/'

    transform = {
        'input': transform_preprocess(config['height_in'], config['width_in']),
        'label': torchvision.transforms.Resize((config['height_out'], config['width_out']),)
    }

    # load dataset
    icdaar2013_dataset_train = SegmentationDataset(
        csv_path=csv_path_train,
        root_input_dir=root_input_dir, input_ext=input_ext,
        root_label_dir=root_label_dir, label_ext=label_ext,
        label_prefix=label_prefix, label_suffix=label_suffix,
        transform=transform
    )
    icdaar2013_dataset_val = SegmentationDataset(
        csv_path=csv_path_val,
        root_input_dir=root_input_dir, input_ext=input_ext,
        root_label_dir=root_label_dir, label_ext=label_ext,
        label_prefix=label_prefix, label_suffix=label_suffix,
        transform=transform
    )
    dataloader_train = DataLoader(dataset=icdaar2013_dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset=icdaar2013_dataset_val, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    best_model = copy.deepcopy(model.state_dict())
    best_loss = None
    # train loop
    print('====START TRAINING====')
    for epoch_idx in range(n_epochs):
        start_time = time.time()
        # scheduler.step(val_epoch_loss)
        # training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for sample in dataloader_train:
            inputs = sample['input'].to(device)  # (m, c, h, w)
            labels = sample['label'].to(device)  # (m, h, w)
            # reset params gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)
                # calculate loss
                loss = criterion(outputs, labels)
                # make prediction
                # output shape (m, c, h, w)
                _, preds = torch.max(outputs, dim=1)
                # backprop + update
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)  # batch loss
            running_corrects = segmentation_correct(preds, labels)
        train_epoch_loss = running_loss / len(icdaar2013_dataset_train)
        train_epoch_acc = running_corrects / len(icdaar2013_dataset_train)

        # validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for sample in dataloader_val:
            inputs = sample['input'].to(device)  # (m, c, h, w)
            labels = sample['label'].to(device)  # (m, h, w)
            # reset params gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                # forward
                outputs = model(inputs)
                # calculate loss
                loss = criterion(outputs, labels)
                # make prediction
                # output shape (m, c, h, w)
                _, preds = torch.max(outputs, dim=1)
            # statistics
            running_loss += loss.item() * inputs.size(0)  # batch loss
            running_corrects = segmentation_correct(preds, labels)
        val_epoch_loss = running_loss / len(icdaar2013_dataset_val)
        val_epoch_acc = running_corrects / len(icdaar2013_dataset_val)

        scheduler.step(val_epoch_loss)

        # use val loss to determine best model
        if best_loss is None:
            best_loss = val_epoch_loss
        if val_epoch_loss <= best_loss:
            best_loss = val_epoch_loss
            best_model = copy.deepcopy(model.state_dict())
            weights_path = os.path.join(
                root_weights_dir, 'Best-Epoch{}-Loss{:.4f}.pt'.format(epoch_idx, val_epoch_loss))
            torch.save(model.state_dict, weights_path)
            print('Best weights saved at {}'.format(weights_path))
        if epoch_idx % save_interval == 0:
            weights_path = os.path.join(
                root_weights_dir, 'E{}_L{:.4f}.pt'.format(epoch_idx, val_epoch_loss))
            torch.save(model.state_dict, weights_path)
            print('Weights saved at {}'.format(weights_path))
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Ep {}/{} - Best loss: {:.4f} - Dur: {:.0f}m {:.0f}s - Train: Loss: {:.4f} Acc: {:.2f}, Val: Loss: {:.4f} Acc: {:.2f}'.format(epoch_idx,
                                                                                                                                            n_epochs - 1,
                                                                                                                                            best_loss,
                                                                                                                                            time_elapsed // 60,
                                                                                                                                            time_elapsed % 60,
                                                                                                                                            train_epoch_loss,
                                                                                                                                            train_epoch_acc,
                                                                                                                                            val_epoch_loss,
                                                                                                                                            val_epoch_acc))
        # print('Train: Loss: {:.4f} Acc: {:.2f}, Val: Loss: {:.4f} Acc: {:.2f}'.format(train_epoch_loss,
        #                                                                               train_epoch_acc,
        #                                                                               val_epoch_loss,
        #                                                                               val_epoch_acc))
    total_training_time = time.time() - start_train_time
    print('=' * 30)
    print('TRAINING COMPLETE! {} epochs in {:.0f}m {:0f}s'.format(n_epochs,
                                                                  total_training_time // 60,
                                                                  total_training_time % 60))
    print('BEST VALIDATION LOSS: {:.4f}'.format(best_loss))

    model.load_state_dict(best_model)
    return model


model_config_path = '/Users/tdh512194/Desktop/Github DL/DeepLearning/Unet/model_config.json'
unet = Unet(config_path=model_config_path, is_training=True).model
optimizer = torch.optim.Adam(unet.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
n_epochs = 6
batch_size = 2
save_interval = 2
train(model=unet, criterion=criterion,
      optimizer=optimizer, scheduler=scheduler,
      n_epochs=n_epochs, batch_size=batch_size,
      save_interval=save_interval, config_path=model_config_path)
