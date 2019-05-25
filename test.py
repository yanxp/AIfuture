# %matplotlib inline
from __future__ import print_function, division
import argparse
import csv
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from PIL import Image
from sklearn import metrics
import VGG_FACE

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str)
    parser.add_argument('--model', dest='model',
                        help='model', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    args = parser.parse_args()
    return args

def cosmetric(galleryFeature, probeFeature):
    metric = []
    for i,p in enumerate(probeFeature):
        vector_a = np.mat(p)
        d = {"value": 0, "index": 0}
        for j,g in enumerate(galleryFeature):
            vector_b = np.mat(g)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            if cos > d["value"]:
                d["value"] = cos
                d["index"] = j

        metric.append(d["index"])

    return metric

def edumetric(galleryFeature, probeFeature, THRESHOD = 1.0):
    LEN_THRESHOD = max(1, int(len(galleryFeature) * 0.25)) # 1 <= x <= 10
    res = []
    for i, p in enumerate(probeFeature):
        metric = np.zeros( (len(galleryFeature),) )
        p = p / np.linalg.norm(p)
        for j, g in enumerate(galleryFeature):
            g = g / np.linalg.norm(g)
            metric[j] = np.sum((p - g) ** 2)
        idx = np.argsort(metric)
        if metric[idx[LEN_THRESHOD]] - metric[idx[0]] >= THRESHOD:
            res.append(idx[0])
        else:
            res.append(-1)
    return res

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])
    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = TripletNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    checkpoint = torch.load(args.model)
    net.load_state_dict(checkpoint)
    net = net.cuda()
    net.eval()

    galleryFeature = []
    probeFeature = []
    ground_truth = []
    rootpath = args.test_dataset
    if rootpath[-1] is not '/':
        rootpath = rootpath + '/'
    gallery = rootpath + "gallery.csv"
    probe = rootpath + "probe.csv"
    img_dir = rootpath + "images/"

    probeFile = open(probe, "r")
    readerProbe = csv.reader(probeFile)
    for item in readerProbe:
        img0_path = img_dir + item[1]
        img0 = Image.open(img0_path).convert("RGB")
        img0 = data_transforms(img0)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        probefeature = net(img0)
        probeFeature.append(probefeature.data.cpu().numpy())
    probeFile.close()

    galleryFile = open(gallery, "r")
    readerGallery = csv.reader(galleryFile)
    for item in readerGallery:
        img1_path = img_dir + item[1]
        img1 = Image.open(img1_path).convert("RGB")
        img1 = data_transforms(img1)
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        galleryfeature = net(img1)
        galleryFeature.append(galleryfeature.data.cpu().numpy())
    galleryFile.close()

    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = list(csv.reader(csvFile))


    for th in [0.3,0.35 ,0.375, 0.4]:
        k = 0
        metric = edumetric(galleryFeature, probeFeature, th)
        #metric = cosmetric(galleryFeature, probeFeature)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
        auc = k / len(metric)
        print('threshod: {} , correct: {}, auc: {}'.format(th, k, auc))
