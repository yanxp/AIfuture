import torch.utils.data as data
from PIL import Image
import os
import os.path as osp
import  random
import numpy as np
class TripletFace(data.Dataset):
    def __init__(self,root,transform=None):
        class_mapping = {cls:i for i,cls in enumerate(os.listdir(root))}
        self.triplets = []
        self.labels = []
        self.transforms = transform
        self.num_class = len(class_mapping)
        threshold = 6
        for k ,cls in enumerate(os.listdir(root)):
            sample_cls = random.sample(os.listdir(root), 1)[0]
            while sample_cls== cls:
                sample_cls = random.sample(os.listdir(root), 1)[0]
            photo = osp.join(root,cls,'1')
            comic = osp.join(root,cls,'0')
            sample_photo = osp.join(root,sample_cls,'1')
            #sample_comic = osp.join(root,sample_cls,'0')
            photo_tmp = os.listdir(photo)
            comic_tmp = os.listdir(comic)
            random.shuffle(photo_tmp)
            random.shuffle(comic_tmp)
            """
            for anchor in photo_tmp[:threshold]:
                for positive in comic_tmp[:threshold]:
                    negative = osp.join(sample_comic,random.sample(os.listdir(sample_comic),1)[0])
                    self.triplets.append((osp.join(photo,anchor),
                                          osp.join(comic,positive),
                                          osp.join(sample_comic ,negative)
                                        ))
                    positive_label = class_mapping[cls]
                    negative_label = class_mapping[sample_cls]
                    self.labels.append((positive_label,negative_label)) 
            """
            for anchor in comic_tmp[:threshold]:
                for positive in photo_tmp[:threshold]:
                    negative = osp.join(sample_photo,random.sample(os.listdir(sample_photo),1)[0])
                    self.triplets.append((osp.join(comic,anchor),
                                         osp.join(photo,positive),
                                         osp.join(sample_photo ,negative)
                                          ))
                    positive_label = class_mapping[cls]
                    negative_label = class_mapping[sample_cls]
                    self.labels.append((positive_label,negative_label))
    
    def __getitem__(self, index):
        anchor,positive,negative = self.triplets[index]
        label_p,label_n = self.labels[index]
        anchor_img = Image.open(anchor).convert('RGB')
        positive_img = Image.open(positive).convert('RGB')
        negative_img = Image.open(negative).convert('RGB')
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)

        return (anchor_img,positive_img,negative_img),(label_p,label_n)


    def __len__(self):
        return  len(self.triplets)
# img_path = '/home/yanxiaopeng/codework/dataset/trainingA/crop_dataset_enlarge_train'
# tripletface = TripletFace(img_path)
# print(len(tripletface))
# print(tripletface[1])
