# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com

import os
import platform
import shutil
import time
import argparse

from PIL import Image
import PIL.ImageOps

import torch
from torchvision import transforms
import torch.nn.functional as F
from numpy import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")


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

def default_loader(path):
    return Image.open(path).convert('RGB')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./My_data/Training_Set/Training', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--model', type=str, default="./checkpoints/classification_efficientnet_b1_1/weights/best.pt")
    parser.add_argument('--image_size', type=int, default=480, help='image_size')
    # parser.add_argument('--model_eca', type=str, default="./checkpoints/newnet_eca_expand_1/weights/1_067_0.8152.pt")
    # parser.add_argument('--model_paddle', type=str, default="./paddle_model/best_model_large_V1.0")
    args = parser.parse_args()
    print(args)

    # Test image folder
    source = args.source + os.sep
    images = os.listdir(source)
    images.sort(key=lambda x: int(x.split('.')[0]))

    # Result list
    result_list_1 = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model, map_location=device)
    print('Load Model Done!!!')
 
    # class_2_index = {0: 'calling', 1: 'normal', 2: 'smoking', 3: 'smoking_calling'}
    # resize = args.image_size + 16
    # crop_size = args.image_size

    file_count = 0
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        for file in images:

            image_path = os.path.join(source, file)

            img0 = default_loader(image_path)

            valid_transform = transforms.Compose([
                transforms.Resize(size=(args.image_size, args.image_size), interpolation=Image.LANCZOS),
                # transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize([0.320, 0.197, 0.103], [0.317, 0.206, 0.128])
            ])

            img = valid_transform(img0)
            img = img.unsqueeze(0)
            model.to(device)
            img = img.to(device)
            out = model(img)
            # preds = F.softmax(out, dim=1) # compute softmax
            # print(preds)
            # prod, index = torch.max(preds, 1)
            pred_labels = torch.sigmoid(out).ge(0.5).float().cpu().numpy()[0]
            # result = process_prediction(sigmoid.cpu().numpy()[0])
            # print(f"{file} -> sigmoid: {list(np.round(sigmoid.cpu().numpy()[0], 2))}")
            print(f"{file} -> result: {pred_labels}")
    
