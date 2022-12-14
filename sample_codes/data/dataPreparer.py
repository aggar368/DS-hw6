from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import os
from utils.options import args
from PIL import Image
import random

class DataPreparation(Dataset):
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None, extra_path=None, 
                 extra_label_path=None, extra_num=None, reduced_num=None):
        
        self.root = root
        self.data_path = data_path 
        self.label_path = label_path 
        
        # Extra data from valid set for train set
        # TODO
        self.extra_path = extra_path
        self.extra_label_path = extra_label_path
        self.extra_num = extra_num
        self.reduced_num = reduced_num

        
        self.transform = transform
        self.target_transform = target_transform
        
        ## preprocess files
        self.preprocess(self.data_path, self.label_path, self.extra_path, 
                        self.extra_label_path, extra_num=self.extra_num, 
                        reduced_num=self.reduced_num)
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        fig_number = int(data_file[:5])
        if fig_number > 45000:
            img_path = os.path.join(self.extra_path, data_file)
        else:
            img_path = os.path.join(self.data_path, data_file)
        image = Image.open(img_path) # plt.imread(img_path)
 
        if self.transform:
            image = self.transform(image)
        
        if self.label_path is None:
            return image, -1, data_file
        
        label = self.file_labels['label'][self.file_labels['image_name'] == data_file].iloc[0]
            
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, data_file
    
    def preprocess(self, data_path, label_path, extra_path, extra_label_path, 
                    extra_num=None, reduced_num=None):
        self.data_files = os.listdir(data_path)
        if reduced_num is not None:   # for valid set
            self.data_files = self.data_files[reduced_num:]

        if extra_path is not None:   # for train set
            extra = os.listdir(extra_path)
            self.data_files = self.data_files + extra[:extra_num]

        self.data_files.sort()
  
        # label-setting part
        if label_path is not None:   # for train&valid set
            self.file_labels = pd.read_csv(label_path)
            if extra_label_path is not None:   # for train set
                extra_labels = pd.read_csv(extra_label_path)
                self.file_labels = pd.concat([self.file_labels, extra_labels[:extra_num]], ignore_index=True)
            else:                              # for valid set
                self.file_labels = self.file_labels[reduced_num:]
                self.file_labels.reset_index(drop=True, inplace=True)
        

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift
        
    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift 
    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)
        
        return img.transform(img.size, Image.AFFINE, (1,0,hshift,0,1,vshift), resample=Image.BICUBIC, fill=1)



class Data:
    def __init__(self, args, data_path, label_path, move):
        

        transform = transforms.Compose([
            transforms.RandomRotation(20), 
            #transforms.RandomHorizontalFlip(p=1.0),
            #transforms.RandomPerspective(p=1.0)
                   #transforms.RandomAdjustSharpness(sharpness_factor=0, p=1.0),
            RandomShift(3),
                   #transforms.GaussianBlur(kernel_size=11)
                  
            
            transforms.Resize((28, 28)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ])
       
        # Validation Data Position
        valid_data_path = data_path.replace('train', 'valid')
        valid_label_path = label_path.replace('train', 'valid')
        
        # Train Data
        train_dataset = DataPreparation(root=args,  
                                        data_path=data_path,
                                        label_path=label_path,
                                        extra_path=valid_data_path,
                                        extra_label_path=valid_label_path,
                                        extra_num=move,
                                        transform=transform)
        
        self.loader_train = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2
            )
        
        
        valid_dataset = DataPreparation(root=args,  
                                       data_path=valid_data_path,
                                       label_path=valid_label_path,
                                       reduced_num=move,
                                       transform=transform)
        
        self.loader_valid = DataLoader(
            valid_dataset, batch_size=args.train_batch_size, shuffle=False, 
            num_workers=2
            )

        # Test Data (No label, only image)
        # TODO
        test_data_path = data_path.replace('train', 'test')

        test_dataset = DataPreparation(root=args, 
                                      data_path=test_data_path,
                                      transform=transform)
        
        self.loader_test = DataLoader(
            test_dataset, batch_size=args.train_batch_size, shuffle=False, 
            num_workers=2
        )