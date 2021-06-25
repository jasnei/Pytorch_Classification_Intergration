# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com
import os
import time
import argparse
from albumentations.augmentations.functional import scale
from albumentations.augmentations.transforms import GaussianBlur, MedianBlur, MotionBlur, RandomBrightness, RandomContrast, ShiftScaleRotate, VerticalFlip
from albumentations.core.composition import OneOf
import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader

from torch_convert import convert_split
from torch_dataset import DatasetFromCSV, MyDataset, AlbumentationsDataset

import albumentations as album
from albumentations.pytorch import ToTensor, ToTensorV2

from torch.utils.tensorboard import SummaryWriter

from models.mobilenet_v3 import MobileNetV3, get_model_parameters
from models.vgg import VGG_11
from models.efficientnet import EfficientNet
from core import core_efficientnet


def plot_loss_acc_curv(train_loss_acc, val_loss_acc, train_label, val_label, ylabel, title, save_dir):
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    plot_range = np.arange(len(train_loss_acc))
    ax1.plot(plot_range, train_loss_acc, label=train_label)
    ax1.plot(plot_range, val_loss_acc, label=val_label)
    ax1.set_title(title)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(which='both', alpha=0.5)
    plt.savefig(save_dir)
    plt.clf()
    plt.close()

def plot_learning_rate_curv(learning_rate_decay, save_dir):
    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(learning_rate_decay, label='learning rate')
    ax3.set_title('Train / Learning_rate: Learning rate decay')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Current lr')
    plt.legend(loc='best')
    plt.grid(which='both', alpha=0.5)
    plt.savefig(save_dir)
    plt.clf()
    plt.close()

class SaveBestModel:
    def __init__(self, epoch, epochs, monitor_value, checkpoint_path, best=None) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.monitor_value = monitor_value
        self.best = best
        self.checkpoin_path = checkpoint_path

    def run(self):
        # Save best model only
        if self.epoch == 0:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = self.monitor_value
            # 每一epoch保存一次
            save_dir = os.path.join(self.checkpoin_path, 'best.pt')
            torch.save(model, save_dir)
            print('saved model')
        elif self.best < self.monitor_value:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = max(self.best, self.monitor_value)
            save_dir = os.path.join(self.checkpoin_path, 'best.pt')
            torch.save(model, save_dir)
            print('saved model')
        elif (self.epoch + 1) == self.epochs:
            save_dir = os.path.join(self.checkpoin_path, 'last.pt')
            torch.save(model, save_dir)
            print('saved model last')
        else:
            pass

def creat_save_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_path = os.path.join(save_path, 'weights')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_path = os.path.join(save_path, 'train_log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path, checkpoint_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./my_data/train_set', help='train_dir')  # train csv
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--n_class', type=int, default=3, help='n_class')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--initial_lr', type=float, default=0.01, help='initial_lr')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--root', type=str, default="./my_data/")
    parser.add_argument('--save_path', type=str, default='./checkpoints/screening_efficientnet_b7_1')
    args = parser.parse_args()
    print(args)

    # Some hyper parameters
    epochs = args.epochs
    n_class = args.n_class
    initial_lr = args.initial_lr
    batch_size = args.batch_size

    # Fist splint train test sets
    convert_split(args.train_dir, train_valid_ratio=0.1, random_state=1)

    # Data root, you have to run torch_convert.py first, then set up the root here
    root = args.root

    # Transforms
    albumentations_train = album.Compose([
        album.Resize(args.image_size, args.image_size),
        album.OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
        album.OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
        album.VerticalFlip(p=0.5),
        album.HorizontalFlip(p=0.5),
        album.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1,
        ),
        album.Normalize(mean=[0.498, 0.527, 0.39], std=[0.227, 0.213, 0.234]),
        ToTensor(),
    ])

    albumentations_valid = album.Compose([
        album.Resize(args.image_size, args.image_size),
        album.Normalize(mean=[0.498, 0.527, 0.39], std=[0.227, 0.213, 0.234]),
        ToTensor(),
    ])

    # Dataset and DataLoader
    train_dataset = AlbumentationsDataset(txt=root + 'train.txt', transform=albumentations_train)
    valid_dataset = AlbumentationsDataset(txt=root + 'valid.txt', transform=albumentations_valid)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=args.num_workers)

    # Model
    # model = EfficientNet.from_name('efficientnet-b7', num_classes=n_class)
    # model = VGG_11(n_class)
    model = MobileNetV3(model_mode="LARGE", num_classes=args.n_class, multiplier=1.0)
    # summary(model, torch.rand(1, 3, 480, 480))
    # print('Total trainable parameters: ', get_model_parameters(model))

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    # loss_fn = torch.nn.BCELoss().to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(lr=initial_lr, params=model.parameters(), momentum=0.9, weight_decay=0) 
    optimizer = torch.optim.Adam(lr=initial_lr, params=model.parameters(), betas=(0.9, 0.99)) 
    # optimizer = torch.optim.RMSprop(lr=initial_lr, params=model.parameters(), alpha=0.9) 
    
    # Learning rate decay
    optimizer_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Warm up 
    # warm_up_epochs = 5
    # warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( np.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * np.pi) + 1)
    # optimizer_step = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # optimizer_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)              # 0.7614
    # optimizer_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    # optimizer_step = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=10, eta_min=1e-5) # 0.8076
    # optimizer_step = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)    
    
    # Weight save path
    save_path = args.save_path
    log_path, checkpoint_path = creat_save_path(save_path)

    log_writer = SummaryWriter(log_path)

    #-----------Training and validation-------
    # For plot training and validation curve
    # train_avg_loss_per_epoch = []
    # train_avg_acc_per_epoch = []
    # val_avg_loss_per_epoch = []
    # val_avg_acc_per_epoch = []
    # learning_rate_decay = []
    # elapse_step = []
    
    save_best_model = SaveBestModel(epoch=None, epochs=args.epochs, monitor_value=None, checkpoint_path=checkpoint_path, best=None)
    train_total = len(train_dataset)
    valid_total = len(valid_dataset)
    
    start = time.time()
    for epoch in range(epochs):
        print('Training......')
        train_acc, train_loss = core_efficientnet.train(model, train_loader, device, loss_fn, optimizer, train_total, epoch)
        print(f'Train -> Epoch: {epoch:>03d}, train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}')
        
        valid_acc, valid_loss = core_efficientnet.valid(model, valid_loader, device, loss_fn, optimizer, valid_total, epoch)
        print(f'Valid -> Epoch: {epoch:>03d}, valid_acc: {valid_acc:.4f}, valid_loss: {valid_loss:.4f}')

        optimizer_step.step()  # 更新学习率
        lr = optimizer.param_groups[0]['lr']
        
        # Save best model only
        save_best_model.epoch = epoch
        save_best_model.monitor_value = valid_acc
        save_best_model.run()

        # Write log
        log_writer.add_scalar("Train/Train_Acc", train_acc, epoch)
        log_writer.add_scalar("Train/Val_Acc", valid_acc, epoch)
        log_writer.add_scalar("Train/Train_Loss", train_loss, epoch)
        log_writer.add_scalar("Train/Val_Loss", valid_loss, epoch)
        log_writer.add_scalar("Train/LR", lr, epoch)

        # # Here plot Accuracy & loss & learning rate curve
        # # Train information
        # train_avg_acc_per_epoch.append(train_acc)
        # train_avg_loss_per_epoch.append(train_loss)
        # learning_rate_decay.append(lr)
        # # Validation information
        # val_avg_acc_per_epoch.append(val_acc)
        # val_avg_loss_per_epoch.append(val_losst)
        # plot_loss_acc_curv(train_avg_acc_per_epoch, val_avg_acc_per_epoch, 'train acc', 'val acc', ylabel='Current Acc', 
        #                     title='Train / Val: Average accuracy per epochs', save_dir=os.path.join(log_path, 'Accuracy_per_epochs.png'))
        # plot_loss_acc_curv(train_avg_loss_per_epoch, val_avg_loss_per_epoch, 'train loss', 'val loss', ylabel='Current Loss', 
        #                     title='Train / Val: Average loss per epochs', save_dir=os.path.join(log_path, 'Loss_per_epochs.png'))
        # plot_learning_rate_curv(learning_rate_decay, save_dir=os.path.join(log_path, 'Learning_rate_per_step.png'))
    
    elapse = time.time() - start
    print(f'Total training time: {elapse}S')