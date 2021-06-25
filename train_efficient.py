# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com
import os
from torch_dataset import *
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader

from core import torch_core_efficient

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from models.newnet_se_expand import get_model_parameters
from loss.focal_loss import FocalLoss

import os 
import warnings
warnings.filterwarnings('ignore')

from models.efficientnet import EfficientNet
from torchsummary import summary

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


if __name__ == '__main__':

    n_class = 3
    # batch_size = 18 # efficientnet-b1
    batch_size = 6  # efficientnet-b5

    resize = 234
    crop_size = 224

    # Data root, you have to run torch_convert.py first, then set up the root here
    root="./my_data/"

    # # Transoforms
    # # 原来用这个可以得到0.8275
    # train_transform = transforms.Compose([
    #     transforms.RandomAffine(20),
    #     transforms.Resize(size=(resize, resize), interpolation=2),
    #     transforms.CenterCrop(size=(crop_size, crop_size)),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    # Expand_3 trial # 0.8035
    transform_color = [transforms.ColorJitter(brightness=0.5),
                       transforms.ColorJitter(contrast=0.5)
                      ]
    transform_affine = [transforms.RandomAffine(0, shear=20),
                        transforms.RandomAffine(0, scale=(0.8, 1.2)),
                        transforms.RandomAffine(0, translate=(0.1, 0.1)),
                        transforms.RandomAffine(degrees=20)]

    train_transform = transforms.Compose([
                    transforms.Resize(size=resize),
                    transforms.RandomApply(transform_affine, p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5), #随机翻转图片
                    transforms.RandomApply(transform_color, p=0.5),
                    transforms.RandomCrop(size=crop_size),
                    transforms.ToTensor(), #将图片变成 Tensor，并且把数值normalize到[0,1]    
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

    valid_transform = transforms.Compose([
        transforms.Resize(size=resize, interpolation=2),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Dataset and DataLoader
    train_dataset = MyDataset(txt=root + 'train.txt', transform=train_transform)
    valid_dataset = MyDataset(txt=root + 'valid.txt', transform=valid_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=4)
    # print(len(train_loader))

    # # 对图像的可视化
    # sample = train_dataset[0]
    # plt.imshow(np.clip(sample[0].permute(1, 2, 0), 0, 1))
    # plt.show()

    # # 测试DataLoader
    # for i, data in train_loader:
    #     print(i.shape)

    model = EfficientNet.from_name('efficientnet-b5', num_classes=n_class)
    # model = darknet.darknet_53(num_classes=n_class)
    print('Total trainable parameters: ', get_model_parameters(model))

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model.to(device)

    # loss function
    # weight = torch.FloatTensor([1.3, 1, 2.8, 1.9]).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # loss_fn = FocalLoss(weight=weight, gamma=1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 下面的参数可以达到了 0.8152 weight_decay=0, initial_lr=0.01, ConsineAnnealingWarmRestarts T_mult=10, CrossEntropy:weight=None, DropOut=0
    # optimizer
    initial_lr = 0.01
    optimizer = torch.optim.SGD(lr=initial_lr, params=model.parameters(), momentum=0.9, weight_decay=0)         # 0.7614
    # opt = torch.optim.RMSprop(lr=initial_lr, params=model.parameters(), alpha=0.9)        # 0.73
    # opt = torch.optim.Adam(lr=initial_lr, params=model.parameters(), betas=(0.9,0.99))    # 0.5187
    # opt_step = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.6)              # 0.7614
    # opt_step = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    # opt_step = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=10, eta_min=1e-5) # 0.8076
    # opt_step = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 20, 30], gamma=0.1) 

    # Train epochs
    epochs = 100
    warm_up_epochs = 5

    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( np.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * np.pi) + 1)
    opt_step = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    
    # Weight save path
    trial = 1
    save_path = 'checkpoints/' + 'efficient_b5_' + str(trial) + os.sep
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_path = save_path + 'weights/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_path = save_path + 'train_log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    #-----------Training and validation-------
    # For plot training and validation curve
    train_avg_loss_per_epoch = []
    train_avg_acc_per_epoch = []
    val_avg_loss_per_epoch = []
    val_avg_acc_per_epoch = []
    learning_rate_decay = []
    elapse_step = []
    
    print('Training......')
    start = time.time()
    for epoch in range(epochs):

        train_acc, train_loss = torch_core_efficient.train(model, train_loader, device, loss_fn, optimizer, epoch)
        # print(np.mean(train_acc), np.mean(train_loss), acc)

        # Train information
        train_avg_acc_per_epoch.append(np.mean(train_acc))
        train_avg_loss_per_epoch.append(np.mean(train_loss))
        
        val_acc_list, val_loss_list, val_acc = torch_core_efficient.valid(model, valid_loader, device, loss_fn, optimizer, epoch)
        # print(np.mean(val_acc), val_acc_total, np.mean(val_loss))

        opt_step.step()  # 更新学习率
        # learning plot
        lr = optimizer.param_groups[0]['lr']
        learning_rate_decay.append(lr)

        # Validation information
        val_avg_acc_per_epoch.append(val_acc)
        val_avg_loss_per_epoch.append(np.mean(val_loss_list))

        # Save best model only
        monitor_val_acc = val_acc
        if epoch == 0:
            best = monitor_val_acc
            # 每一epoch保存一次
            save_dir = os.path.join(checkpoint_path, f'{trial}_{epoch:03d}_{best:.4f}.pt')
            torch.save(model, save_dir)
            print('saved model')
        elif best < monitor_val_acc:
            print(f'val_acc: {monitor_val_acc:.4f}')
            best = max(best, monitor_val_acc)
            save_dir = os.path.join(checkpoint_path, f'{trial}_{epoch:03d}_{best:.4f}.pt')
            torch.save(model, save_dir)
            print('saved model')
        else:
            pass
        
        # Here plot Accuracy & loss & learning curve
        plot_loss_acc_curv(train_avg_acc_per_epoch, val_avg_acc_per_epoch, 'train acc', 'val acc', ylabel='Current Acc', title='Train / Val: Average accuracy per epochs', save_dir=log_path + f'{trial}_Accuracy_per_epochs.png')
        plot_loss_acc_curv(train_avg_loss_per_epoch, val_avg_loss_per_epoch, 'train loss', 'val loss', ylabel='Current Loss', title='Train / Val: Average loss per epochs', save_dir=log_path + f'{trial}_Loss_per_epochs.png')
        plot_learning_rate_curv(learning_rate_decay, save_dir=log_path + f'{trial}_Learning_rate_per_step.png')
    
    elapse = time.time() - start
    print(f'Total training time: {elapse}S')