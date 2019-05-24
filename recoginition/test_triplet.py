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

from sklearn import metrics
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

VGG_FACE = nn.Sequential(  # Sequential,
    nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    Lambda(lambda x: x.view(x.size(0), -1)),  # View,
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(25088, 4096)),  # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 4096)),  # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 2622)),  # Linear,
    nn.Softmax(),
)


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

def cos_sim(Pfeature , Cfeature):
    metric = []
    for i in range(len(Pfeature)):
        vector_a = np.mat(Pfeature[i])
        vector_b = np.mat(Cfeature[i])
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        if cos < 0.55 :
            cos = 0.0
        if cos > 0.9 :
            cos = 1.0
        metric.append(cos)
    return metric


def calEuclideanDistance(Pfeature, Cfeature):
    metric = []
    
    for i in range(len(Pfeature)):
        m = np.linalg.norm(Pfeature[i] - Cfeature[i])
        metric.append(m)
    sort_metric = sorted(metric)
    a = np.mean(sort_metric[:10])
    b = np.mean(sort_metric[-10:])
    for i in range(len(metric)):
        if metric[i]>b:
            metric[i] = 0.0
        elif metric[i]<a:
            metric[i] = 1.0
        else:
            metric[i] = (b - metric[i]) / (b - a)
     
    # a = min(metric)
    # b = max(metric)
    # for i in range(len(metric)):
    #     metric[i] = (b-metric[i]) / (b-a)
    
    return metric


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = VGG_FACE
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

def aucfun(act,pred):
    fpr , tpr ,thresholds = metrics.roc_curve(act , pred , pos_label= 1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

if __name__ == '__main__':
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = SiameseNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    checkpoint = torch.load(args.model)
    net.load_state_dict(checkpoint)
    print('load model successfully!')
    net = net.cuda()
    net.eval()

    Pfeature = []
    Cfeature = []
    ground_truth = []
    rootpath = args.test_dataset
    if rootpath[-1] is not '/':
        rootpath = rootpath + '/'
    data_file = rootpath + "list.csv"
    data_rpath = rootpath
    csvFile = open(data_file, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        img0_path, img1_path = data_rpath + item[1], data_rpath + item[2]
        ground_truth.append(int(item[3]))
        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")
        img0 = data_transforms(img0)
        img1 = data_transforms(img1)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        with torch.no_grad():
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
    metric = calEuclideanDistance(photo_feature, cari_feature)
    auc = aucfun(ground_truth, metric)
    print('auc:',auc)
    filename = 'predictions.csv'
    ppath = args.ppath
    if ppath is not None:
       if not os.path.exists(ppath):
           os.makedirs(ppath)
       if ppath[-1] is not '/':
            ppath = ppath + '/'
       filename = ppath + filename
    csvFile = open(filename, 'w')
    fileHeader = ["group_id", "confidence","auc"]
    writer = csv.writer(csvFile)
    writer.writerow(fileHeader)
    for i in range(len(metric)):
        writer.writerow([i + 1, metric[i], auc])
    csvFile.close()
