# !/usr/bin/python
# -*- coding: utf-8 -*-
# jasnei@163.com

import os
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def convert_to_img(root, train_val, save_file_name, train=True):
    if(train):
        f=open(root + save_file_name,'w')
        data_path=root + train_val
        folder_list = os.listdir(data_path)
        # print(folder_list)
        data = []
        for index in range(len(folder_list)):
            folder = folder_list[index]
            # print(folder)
            for root, dirs, files in os.walk(os.path.join(data_path, folder)):
                # print(root)
                for file in files:
                    image_path = root + '/'+ file  # os.path.join(root, file)
                    data.append([image_path, index])
                    f.write(image_path + ' ' + str(index))
                    f.write('\n')
        print(f'There are {len(data)} images for training.')
        f.close()
    else:
        f=open(root+ save_file_name, 'w')
        data_path=root + train_val
        folder_list = os.listdir(data_path)
        data = []
        for index in range(len(folder_list)):
            folder = folder_list[index]
            for root, dirs, files in os.walk(os.path.join(data_path, folder)):
                for file in files:
                    image_path = root + '/'+ file  # os.path.join(root, file)
                    data.append([image_path, index])
                    f.write(image_path + ' ' + str(index))
                    f.write('\n')
        print(f'There are {len(data)} images for validation.')
        f.close()

def convert_label(image_csv, label, data_root, train_txt, valid_txt, train_valid_ratio):
    """
    image_csv: path for the image csv
    label: label path
    train_test_ratio: percentage of train_test, 0.1 means 0.1 for test, 0.9 for train
    """
    # load label.txt create label dict to encode the label
    label_dict = dict()
    with open(label, encoding='utf-8') as file:
        for i, name in enumerate(file.readlines()):
            label_dict[name.rstrip("\n")] = i
    # print(label_dict)
    
    # collect all the data 
    data_total = []
    with open(image_csv, 'r', encoding='utf-8') as file:
        for line in file.readlines()[1:]:
            name = line.split(",")
            image_path = name[0]
            image_name = name[1].rstrip("\n")
            data_total.append(data_root + image_path + ' ' + str(label_dict[image_name]))
            # print(label_dict[image_name])
        
    train, test = train_test_split(data_total, test_size=train_valid_ratio)

    ftrain = open(train_txt, 'w')
    for i in train:
        ftrain.write(i)
        ftrain.write('\n')
    ftrain.close

    ftest = open(valid_txt, 'w')
    for i in test:
        ftest.write(i)
        ftest.write('\n')
    ftest.close
    print(f"There are {len(data_total)} images, {len(train)} for training, {len(test)} for validation")

def convert_split(train_dir, train_valid_ratio=0.1, random_state=0):

    data_path=train_dir
    txt_root = os.path.split(train_dir)[0]
    txt_root_files = os.listdir(txt_root)
    if "train.txt" and "valid.txt" in txt_root_files:
        print("train test already split")
        return
    else:
        folder_list = os.listdir(data_path)
        # print(folder_list)
        # 先判断是否是文件夹，并排序
        folders = []
        for index in range(len(folder_list)):
            if os.path.isdir(os.path.join(data_path, folder_list[index])):
                folders.append(folder_list[index])
        folders.sort(key=lambda x: int(x), reverse=False)

        data_total = []
        for index in range(len(folders)):
            folder = folders[index]
            folder = os.path.join(data_path, folder)
            if os.path.isdir(folder):    
                for root, dirs, files in os.walk(folder):
                    # print(root)
                    for file in files:
                        image_path = os.path.join(root, file) #root + os.sep + file 
                        data_total.append(image_path + ' ' + str(index))

        # train test split
        train, test = train_test_split(data_total, test_size=train_valid_ratio, random_state=random_state)

        ftrain = open(os.path.join(txt_root, "train.txt"), 'w', encoding='utf-8')
        for i in train:
            ftrain.write(i)
            ftrain.write('\n')
        ftrain.close

        ftest = open(os.path.join(txt_root, "valid.txt"), 'w', encoding='utf-8')
        for i in test:
            ftest.write(i)
            ftest.write('\n')
        ftest.close
        print(f"There are {len(data_total)} images, {len(train)} for training, {len(test)} for validation")

def conver_split_csv(csv_path, train_valid_ratio=0.1, random_state=0):
    
    path_splits = csv_path.split("/")
    save_path = ""
    for split in path_splits[:-1]:
        save_path = os.path.join(save_path, split)

    save_path_files = os.listdir(save_path)
    if "train.csv" and "valid.csv" in save_path_files:
        print("Train Test already split up!")
        return
    else:
        df = pd.read_csv(csv_path)
        train, valid = train_test_split(df, test_size=train_valid_ratio, random_state=random_state)        
        train.to_csv(os.path.join(save_path, "train.csv"), index=False)
        valid.to_csv(os.path.join(save_path, "valid.csv"), index=False)
        print(f"Split Done! There are {len(train)} for training, {len(valid)} for validation!")

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./my_data/train_set', help='train_dir')  # train data path
    args = parser.parse_args()
    print(args)
    convert_split(args.train_dir, train_valid_ratio=0.1, random_state=1)

    # label = 'label.txt'
    # data_root = "./data/input/ButterflyClassification/"
    # image_csv = "./data/input/ButterflyClassification/train.csv"
    # train_txt = data_root + "train.txt"
    # valid_txt = data_root + "valid.txt"

    # convert_label(image_csv, label, data_root, train_txt, valid_txt, train_valid_ratio=0.1)

    # root="./data/input/ButterflyClassificatoin/image/"
    # convert_to_img('./my_data/', 'train_1/', 'train_1.txt', True)
    # convert_to_img('./my_data/', 'valid_1/', 'valid_1.txt',False)

    # root="./my_data/"
    # convert_to_img('./my_data/', 'train/', 'train.txt', True)
    # convert_to_img('./my_data/', 'valid/', 'valid.txt', False)

    # conver_split_csv("My_data/Train_set/training_set.csv", train_valid_ratio=0.1)