# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com
import os

import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img

def change_2848_loader(path):
    img = Image.open(path).convert('RGB')
    
    if img.size[0] < 2800:
        return img
    else:
        img_pad = np.zeros((img.size[1] + 800, img.size[0], 3), dtype=np.uint8)
        img_pad[400:400+2848, 160:] = np.asanyarray(img)[:, :-160]
        img = Image.fromarray(img_pad)
        return img

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class DatasetFromCSV(Dataset):
    def __init__(self, image_root, csv_path, transforms=None, loader=default_loader):
        
        self.image_root = image_root
        
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 1])
        
        # process image name
        imgs = []
        files_names = np.array(self.data.iloc[:, 0])
        for img in files_names:
            imgs.append(os.path.join(self.image_root, str(img)+'.png'))
        
        self.images = imgs
        self.transforms = transforms
        self.loader = loader
 
    def __getitem__(self, index):

        label = self.labels[index]
        img = self.images[index]
        img = self.loader(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    
    def __len__(self):
        return len(self.data.index)

class DatasetFromCSVClass(Dataset):
    def __init__(self, image_root, csv_path, transforms=None, loader=change_2848_loader):
        
        self.image_root = image_root
        
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 2:])
        
        # process image name
        imgs = []
        files_names = np.array(self.data.iloc[:, 0])
        for img in files_names:
            imgs.append(os.path.join(self.image_root, str(img)+'.png'))
        
        self.images = imgs
        self.transforms = transforms
        self.loader = loader
 
    def __getitem__(self, index):

        label = self.labels[index]
        img = self.images[index]
        img = self.loader(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    
    def __len__(self):
        return len(self.data.index)

class AlbumentationsDataset(Dataset):
    """ 
        处理数据增强跟上面的 TorchvisionDataset 的一致
    """
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)

        if self.transform is not None:
            image_np = np.array(img)
            augmented = self.transform(image=image_np)
            # img = Image.fromarray(augmented['image'])
            img = augmented['image']
        return img,label

    def __len__(self):
        return len(self.imgs)

        
if __name__ == '__main__':
    pass

