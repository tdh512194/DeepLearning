import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from .dataloader import SegmentationDataset
from .utils import transform_preprocess, is_correct
from .Unet import Unet

def train(model, criterion, optimizer, scheduler, n_epochs, batch_size, save_interval, config_path):
    start_train_time = time.time()
    # define training configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    csv_path_train = ''
    csv_path_val = ''
    root_input_dir = ''
    input_ext = ''
    root_label_dir = ''
    label_ext = ''
    label_prefix = ''
    label_suffix = ''

    root_weights_dir = ''

    transform = {
        'input': transform_preprocess(config['height_in'], config['width_in']),
        'label': torchvision.transforms.Resize(config['height_out'], config['width_out'])
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
    dataloader_train = DataLoader(datasets=icdaar2013_dataset_train, batch_size=batch_size, 
                                  shuffle=True, num_workers=2)
    dataloader_val = DataLoader(datasets=icdaar2013_dataset_val, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
    
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    # train loop
    for epoch_idx in range(n_epochs):
        start_time = time.time()
        scheduler.step()
        # training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for sample in dataloader_train:
            inputs = sample['input'].to(device) # (m, c, h, w)
            labels = sample['label'].to(device) # (m, h, w)
            # reset params gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)
                # calculate loss
                loss = criterion(outputs, labels)
                # make prediction
                _, preds = torch.max(outputs, dim=1) # output shape (m, c, h, w)
                # backprop + update
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0) # batch loss
            running_corrects = is_correct(preds, labels)
        train_epoch_loss = running_loss / len(icdaar2013_dataset_train)
        train_epoch_acc = running_corrects.double() / len(icdaar2013_dataset_train)
        print('TRAIN: Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, train_epoch_acc)) 
        
        # validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for sample in dataloader_val:
            inputs = sample['input'].to(device) # (m, c, h, w)
            labels = sample['label'].to(device) # (m, h, w)
            # reset params gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                # forward
                outputs = model(inputs)
                # calculate loss
                loss = criterion(outputs, labels)
                # make prediction
                _, preds = torch.max(outputs, dim=1) # output shape (m, c, h, w)
            # statistics
            running_loss += loss.item() * inputs.size(0) # batch loss
            running_corrects = is_correct(preds, labels)
        val_epoch_loss = running_loss / len(icdaar2013_dataset_val)
        val_epoch_acc = running_corrects.double() / len(icdaar2013_dataset_val)
        print('VAL: Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
        # use val loss to determine best model
        if val_epoch_loss <= best_loss:
            best_loss = val_epoch_loss
            best_model = copy.deepcopy(model.state_dict())
            weights_path = os.path.join(root_weights_dir, 'Best-Epoch{}-Loss{:.4f}.pt'.format(epoch_idx, val_epoch_loss))
            torch.save(model.state_dict, weights_path)
            print('Best weights saved at {}'.format(weights_path))
        if epoch_idx % save_interval == 0:
            weights_path = os.path.join(root_weights_dir, 'Epoch{}-Loss{:.4f}.pt'.format(epoch_idx, val_epoch_loss))
            torch.save(model.state_dict, weights_path)
            print('Weights saved at {}'.format(weights_path))
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Epoch {}/{}, Best loss: {:.4f} - {:.0f}m {:.0f}s'.format(epoch_idx,
                                                                        n_epochs - 1,
                                                                        best_loss,
                                                                        time_elapsed // 60, 
                                                                        time_elapsed % 60))
        print('TRAIN: Loss: {:.4f} Acc: {:.4f}, VAL: Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss,
                                                                                    train_epoch_acc,
                                                                                    val_epoch_loss,
                                                                                    val_epoch_acc))
    total_training_time = time.time() - start_train_time
    print('*' * 10)
    print('TRAINING COMPLETE! {} epochs in {:.0f}m {:0f}s'.format(n_epochs, 
                                                                start_train_time // 60,
                                                                total_training_time % 60))
    print('BEST VALIDATION LOSS: {:.4f}'.format(best_loss))

    model.load_state_dict(best_model)
    return model


        