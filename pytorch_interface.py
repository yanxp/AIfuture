# -*- coding:utf-8 -*-
from __future__ import print_function, division
import os
from PIL import Image
import cv2
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import VGG_FACE
from face_model import FaceModel
from retinaface import RetinaFace

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])
        
    def forward(self, x):
        out = self.model(x)
        out = F.normalize(out, p=2, dim=1)
        return out

def edumetric(galleryFeature, probeFeature, THRESHOD = 0.166):
    LEN_THRESHOD = max(1, int(len(galleryFeature) * 0.25)) # 1 <= x <= 10
    res = []
    for i, p in enumerate(probeFeature):
        metric = np.zeros( (len(galleryFeature),) )
        # p = p / np.linalg.norm(p)
        for j, g in enumerate(galleryFeature):
            # g = g / np.linalg.norm(g)
            metric[j] = np.sum((p - g) ** 2)
        idx = np.argsort(metric)
        if metric[idx[LEN_THRESHOD]] - metric[idx[0]] >= THRESHOD:
            res.append(idx[0])
        else:
            res.append(-1)
    return res

def detect_or_return_origin(img_path, model):
    img = cv2.imread(img_path)
    new_img = model.get_input(img, threshold=0.02)

    if new_img is None:
        img = cv2.resize(img, (256, 256))
        b = (256 - 224) // 2
        img = img[b:-b, b:-b, :]
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        new_img = cv2.resize(new_img, (224, 224))
        return Image.fromarray(new_img)

def predict_interface(imgset_rpath: str, gallery_dict: dict, probe_dict: dict) -> [(str, str), ...]:
    # 1. load model and other settings
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    net = TripletNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    load_name = os.getenv('PRETRAINED_MODEL')
    checkpoint = torch.load(load_name)
    checkpoint = {k: checkpoint[k] for k in net.state_dict().keys() }
    net.load_state_dict(checkpoint)
    net = net.cuda()
    net.eval()

    detector = RetinaFace("./models/testR50", 4, 0, 'net3', 0.4, False, vote=False)
    fmodel = FaceModel(detector)
    # 2. get features
    probe_list = [(k, v) for k, v in probe_dict.items()]
    gallery_list = [(k, v) for k, v in gallery_dict.items()]
    galleryFeature = []
    probeFeature = []
    prob_imgs = []
    gallery_imgs = []
    for _, item in probe_list:
        img0_path = os.path.join(imgset_rpath, item)
        img0 = detect_or_return_origin(img0_path, fmodel)
        prob_imgs.append(img0)

    for _, item in gallery_list:
        img1_path = os.path.join(imgset_rpath, item)
        img1 = detect_or_return_origin(img1_path, fmodel)
        gallery_imgs.append(img1)
    del detector

    for img0 in prob_imgs:
        img0 = data_transforms(img0)
        img0 = Variable(img0.unsqueeze(0)).cuda()
        probefeature = net(img0)
        probeFeature.append(probefeature.data.cpu().numpy())

    for img1 in gallery_imgs:
        img1 = data_transforms(img1)
        img1 = Variable(img1.unsqueeze(0)).cuda()
        galleryfeature = net(img1)
        galleryFeature.append(galleryfeature.data.cpu().numpy())

    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    preds = edumetric(galleryFeature, probeFeature)

    # 3. prepare result
    result = [] # result = [("1", "2"), ("2", "4")]
    for i, p in enumerate(preds):
        if p != -1:
            result.append((probe_list[i][0], gallery_list[p][0]))
        else:
            result.append((probe_list[i][0], "-1"))

    return result