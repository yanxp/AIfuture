import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import pickle
from random import shuffle
import torch
import os,random
import collections
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]

def make_triplet_dataset(dir,x):
    data_root = os.path.join(dir, 'data_triples_'+x+'.pkl')
    images = pickle.load(open(data_root, 'rb'))
    return images

class TripletFace(Dataset):
    def __init__(self,img_path,x,transform=None):
        class_mapping = {cls:i for i,cls in enumerate(os.listdir(os.path.join(img_path,'images')))}
        self.labels = []
        self.triplets = make_triplet_dataset(img_path, x)
        for triplet in self.triplets:
            self.labels.append(class_mapping[triplet[0].split('/')[-3]])
        self.transforms = transform

    def __getitem__(self, index):
        anchor,positive,negative = self.triplets[index]
        anchor_img = Image.open(anchor).convert('RGB')
        positive_img = Image.open(positive).convert('RGB')
        negative_img = Image.open(negative).convert('RGB')
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)

        return (anchor_img,positive_img,negative_img),self.labels[index]


    def __len__(self):
        return  len(self.triplets)

