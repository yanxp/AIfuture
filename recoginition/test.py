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

def cos_sim(P , C):
    metric = []
    for i in range(len(C)):
        vector_a = np.mat(C[i])
        d = {"max": 0.0, "index": 0}
        for j in range(len(P)):
            vector_b = np.mat(P[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            if cos < 0.55 :
                cos = 0.0
            if cos > 0.9 :
                cos = 1.0
            if cos > d["max"]:
                d["max"] = cos
                d["index"] = j

        metric.append(d["index"])
    return metric


def calEuclideanDistance(Pfeature , Cfeature):
    metric = []
    for i in range(len(Pfeature)):
        m = np.linalg.norm( Pfeature[i] - Cfeature[i])
        metric.append(m)
    a = min(metric)
    b = max(metric)
    for i in range(len(metric)):
        metric[i] = (b - metric[i])/(b - a)
    return metric

def distance(P , C):
    metric = []
    for i in range(len(C)):
        vector_a = np.mat(C[i])
        d = {"max": 0.0, "index": 0}
        for j in range(len(P)):
            vector_b = np.mat(P[j])
            # num = float(vector_a * vector_b.T)
            # denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            distance = np.linalg.norm(vector_a - vector_b)
            # if cos < 0.55 :
            #     cos = 0.0
            # if cos > 0.9 :
            #     cos = 1.0
            if distance < d["max"]:
                d["max"] = distance
                d["index"] = j

        metric.append(d["index"])
    return metric

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

VGG_FACE = nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    Lambda(lambda x: x.view(x.size(0),-1)), # View,
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
    nn.Softmax(),
)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.cnn = VGG_FACE
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1):
        output1 = self.forward_once(input1)
        return output1

if __name__ == '__main__':
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = Network()
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
    person_file = rootpath + "gallery.csv"
    cari_file = rootpath + "probe.csv"
    # data_rpath = rootpath + "images/"
    data_rpath = rootpath 
    
    cariFile = open(cari_file, "r")
    readerC = csv.reader(cariFile)
    for item in readerC:
        # if readerC.line_num == 1:
        #     continue
        img0_path = data_rpath + item[1]
        img0 = Image.open(img0_path).convert("RGB")
        img0 = data_transforms(img0)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        output1 = net(img0)
        Vis1 = output1.data.cpu().numpy()
        for j in range(Vis1.shape[0]):
            B = np.divide(Vis1[j], np.linalg.norm(Vis1[j], ord=2))
            Cfeature.append(B.tolist())
    cariFile.close()

    personFile = open(person_file, "r")
    readerP = csv.reader(personFile)
    for item in readerP:
        # if readerP.line_num == 1:
        #     continue
        img1_path = data_rpath + item[1]
        img1 = Image.open(img1_path).convert("RGB")
        img1 = data_transforms(img1)
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        output2 = net(img1)
        Vis2 = output2.data.cpu().numpy()
        for j in range(Vis2.shape[0]):
            B = np.divide(Vis2[j], np.linalg.norm(Vis2[j], ord=2))
            Pfeature.append(B.tolist())
    personFile.close()

    photo_feature = np.array(Pfeature)
    cari_feature = np.array(Cfeature)
    metric = cos_sim(photo_feature, cari_feature)
    # metric = distance(photo_feature, cari_feature)

    filename = 'predictions.csv'
    ppath = args.ppath
    if ppath is not None:
       if not os.path.exists(ppath):
           os.makedirs(ppath)
       if ppath[-1] is not '/':
            ppath = ppath + '/'
       filename = ppath + filename
    csvFile = open(filename, 'w')
    # fileHeader = ["group_id", "confidence"]
    writer = csv.writer(csvFile)
    # writer.writerow(fileHeader)
    for i in range(len(metric)):
        writer.writerow([i, metric[i]])
    csvFile.close()

    k = 0
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = csv.reader(csvFile)

    for item in readerC:
        if metric[int(item[0])] == int(item[1]):
            k += 1

    auc = k / len(metric)
    print(k)
    print(auc)
