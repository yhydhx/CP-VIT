from __future__ import print_function
from re import L
import re
from parameters import *
import torch.utils.data as data
import random
import os
import numpy as np
import torch
from sampler import BalancedBatchSampler
import albumentations as A
import cv2
import pdb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import matplotlib.pyplot as plt


def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img


class CP_Net_Dataset(data.Dataset):
    def __init__(self, data_training, data_testing, train=True, test=False, img_resize=224):
        self.train = train  # training set or val set
        self.test = test
        self.img_resize = img_resize

        # pdb.set_trace()
        if self.train:
            self.data = data_training
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        if self.test:
            self.data = data_testing
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        random.shuffle(self.data)


    def __getitem__(self, index):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        #resized_img = img_resize(img, self.img_resize)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        return transformed_img, label_ID

    def debug_getitem__(self, index=0):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        # if not self.train and not self.test:
        #     print(img_path)
        #     print(img.shape)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_orig_(' + str(img.shape[0]) + '_' + str(img.shape[1]) + ')' + '.jpg', img)
        #resized_img = img_resize(img, self.img_resize)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_resize_(' + str(resized_img.shape[0]) + '_' + str(resized_img.shape[1]) + ')' + '.jpg', resized_img)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_transformed_(' + str(transformed_img.shape[0]) + '_' + str(transformed_img.shape[1]) + ')' + '.jpg', cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        print(transformed_img.shape)
        #pdb.set_trace()

        return img_path, transformed_img, label_ID

    def __len__(self):
        return len(self.data)

def get_cal_loader():
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    trainset = datasets.Caltech256(root="./data",
                                    download=True,
                                    transform=transform_train
                                )
    testset = datasets.Caltech256(root="./data",
                                   download=True,
                                   transform=transform_test
                                   ) 
    
    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.batch_size,
                             num_workers=4,
                             ) if testset is not None else None
    return train_loader, test_loader

def get_imagenet(root, target_transform = None):
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tra_root = os.path.join(root,'train')
        trainset = datasets.ImageFolder(root=tra_root,
                                transform=False,
                                target_transform=None)
        val_root = os.path.join(root,'val')
        valset = datasets.ImageFolder(root=val_root,
                                transform=False,
                                target_transform=None)
        return trainset,valset


def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.batch_size,
                             num_workers=4,
                             ) if testset is not None else None
    return train_loader, test_loader


if __name__ == '__main__':

    root_path = opt.imagenet_path
    train_loader, test_loader = get_cal_loader()
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()



