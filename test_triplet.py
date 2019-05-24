# %matplotlib inline
from __future__ import print_function, division

import argparse

import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F
import math
import VGG_FACE
from sklearn import metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Triplet network')
    parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str)
    parser.add_argument('--model', dest='model',
                        help='the model path', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    args = parser.parse_args()
    return args

def calEuclideanDistance(Pfeature, Cfeature):
    metric = []
    for i in range(len(Pfeature)):
        m = np.linalg.norm(Pfeature[i] - Cfeature[i])
        metric.append(m)
    sort_metric = sorted(metric)
    a = np.mean(sort_metric[:35])
    b = np.mean(sort_metric[-35:])
    for i in range(len(metric)):
        if metric[i]>b:
            metric[i] = 0.0
        elif metric[i]<a:
            metric[i] = 1.0
        else:
            metric[i] = (b - metric[i]) / (b - a)
    return metric

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

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

    Pfeature = []
    Cfeature = []
    ground_truth = []
    rootpath = args.test_dataset
    if rootpath[-1] is not '/':
        rootpath = rootpath + '/'
    data_file = rootpath + "list.csv"
    data_rpath = rootpath + "images/"
    csvFile = open(data_file, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        img0_path, img1_path = data_rpath + item[1], data_rpath + item[2]
        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")
        img0 = data_transforms(img0)
        img1 = data_transforms(img1)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        output1, output2, _= net(img0, img1,img0)

        Vis1 = output1.data.cpu().numpy()
        for j in range(Vis1.shape[0]):
            B = np.divide(Vis1[j], np.linalg.norm(Vis1[j], ord=2))
            Pfeature.append(B.tolist())

        Vis2 = output2.data.cpu().numpy()
        for j in range(Vis2.shape[0]):
            B = np.divide(Vis2[j], np.linalg.norm(Vis2[j], ord=2))
            Cfeature.append(B.tolist())

    csvFile.close()
    photo_feature = np.array(Pfeature)
    cari_feature = np.array(Cfeature)
    Eucmetric = calEuclideanDistance(photo_feature, cari_feature)
    filename = 'predictions.csv'
    ppath = args.ppath
    if ppath is not None:
       if not os.path.exists(ppath):
           os.makedirs(ppath)
       if ppath[-1] is not '/':
            ppath = ppath + '/'
       filename = ppath + filename
    csvFile = open(filename, 'w')
    fileHeader = ["group_id", "confidence"]
    writer = csv.writer(csvFile)
    writer.writerow(fileHeader)
    for i in range(len(Eucmetric)):
        writer.writerow([i + 1, Eucmetric[i]])
    csvFile.close()
    print('done!')
