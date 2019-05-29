# %matplotlib inline
from __future__ import print_function, division

import torchfile as torchfile
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable,Function
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from torchvision.models import resnet18
from dataloader import TripletFace
import VGG_FACE
from sklearn import metrics
from torch.optim import lr_scheduler
import math
import VGG_FACE
import adabound
from VGG_FACE import load_weights
from metrics import *

class TripletNetwork(nn.Module):
    def __init__(self,num_class):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        load_weights(self.cnn, 'VGG_FACE.t7')
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, num_class))
    def forward(self, x1,x2,x3):
        feat1 = self.model(x1)
        feat2 = self.model(x2)
        feat3 = self.model(x3)
        #res1 = F.sigmoid(self.classifier(torch.cat((feat1,feat2),dim=1)))
        #res2 = F.sigmoid(self.classifier(torch.cat((feat1, feat3), dim=1)))
        res1 = self.classifier(feat1)
        res2 = self.classifier(feat2)
        res3 = self.classifier(feat3)
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        feat3 = F.normalize(feat3, p=2, dim=1)
        return feat1,feat2,feat3,res1,res2,res3

class TripletLoss(nn.Module):
    """
    Triplet loss
    """
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def train(model,criterion,loss_fn,optimizer, epoch,scheduler=None):
    if scheduler is not None:
        scheduler.step()
        model.train(True)  # Set model to training mode
    else:
        model.train(True)  # Set model to training mode
    losses = []
    for batch_idx, ((data_a, data_p, data_n),(label_p,label_n)) in enumerate(train_loaders):
        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a,requires_grad=True), Variable(data_p,requires_grad=True), \
                                 Variable(data_n,requires_grad=True)
        label_p,label_n = Variable(label_p),Variable(label_n)
        optimizer.zero_grad()
        # compute output
        out_a, out_p, out_n, res1, res2, res3 = model(data_a,data_p,data_n)
        # Choose the hard negatives
        d_p = F.pairwise_distance(out_a, out_p)
        d_n = F.pairwise_distance(out_a, out_n)
        """
        all = (d_n - d_p < margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue
        out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)
        out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)
        out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)

        selected_data_a = Variable(torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)
        selected_data_p = Variable(torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)
        selected_data_n = Variable(torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).cuda(),requires_grad=True)
        
        selected_label_p = Variable(torch.from_numpy(label_p.cpu().numpy()[hard_triplets])).cuda()
        selected_label_n= Variable(torch.from_numpy(label_n.cpu().numpy()[hard_triplets])).cuda()
        selected_res1= Variable(torch.from_numpy(res1.cpu().data.numpy()[hard_triplets])).cuda()
        selected_res2= Variable(torch.from_numpy(res2.cpu().data.numpy()[hard_triplets])).cuda()
        selected_res3= Variable(torch.from_numpy(res3.cpu().data.numpy()[hard_triplets])).cuda()

        
        triplet_loss = criterion(selected_data_a, selected_data_p, selected_data_n)
        #triplet_loss = criterion(out_a, out_p, out_n)

        # _,cls_a = model(selected_data_a)
        # _,cls_p = model(selected_data_p)
        # _,cls_n = model(selected_data_n)
        """
        #predicted_labels = torch.cat([res1,res2])
        #true_labels = torch.cat([Variable(torch.ones(res1.size()).cuda()),Variable(torch.zeros(res2.size()).cuda())])
        
        predicted_labels = torch.cat([res1,res2,res3])
        true_labels = torch.cat([label_p,label_p,label_n])
        triplet_loss = criterion(out_a, out_p, out_n)
        
        #predicted_labels = torch.cat([selected_res1,selected_res2,selected_res3])
        #true_labels = torch.cat([selected_label_p,selected_label_p,selected_label_n])

        bce_loss = loss_fn(predicted_labels.cuda(),true_labels.cuda())

        loss = bce_loss + triplet_loss
        # compute gradient and update weights
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('epoch:{},loss:{}'.format(epoch,np.mean(losses)))
    return model

def cosmetric(galleryFeature, probeFeature):
    metric = []
    totalnum = collections.defaultdict(int)
    threshold = 142
    for i,p in enumerate(probeFeature):
        vector_a = np.mat(p)
        d = {"value": 0, "index": 0, "second":0}
        for j,g in enumerate(galleryFeature):
            vector_b = np.mat(g)
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            if cos > d["value"] and totalnum[j]<threshold:
                totalnum[j] += 1
                d["second"] = d["index"]
                d["value"] = cos
                d["index"] = j
        """
        totalnum[d["index"]] += 1
        if totalnum[d["index"]]<threshold:
            metric.append(d["index"])
        else:
            metric.append(d["second"])
        """
        metric.append(d["index"])

    return metric

    
def test(net,testpath):
    net.eval()
    galleryFeature = []
    probeFeature = []
    ground_truth = []
    rootpath = testpath
    if rootpath[-1] is not '/':
        rootpath = rootpath + '/'
    gallery = rootpath + "gallery.csv"
    probe = rootpath + "probe.csv"
    img_dir = rootpath 

    probeFile = open(probe, "r")
    readerProbe = csv.reader(probeFile)
    for item in readerProbe:
        img0_path = img_dir + item[1]
        img0 = Image.open(img0_path).convert("RGB")
        img0 = data_transforms(img0)
        with torch.no_grad():
            img0 = Variable(img0.unsqueeze(0)).cuda()
        probefeature = net(img0,img0,img0)[0]
        probeFeature.append(probefeature.data.cpu().numpy())
    probeFile.close()

    galleryFile = open(gallery, "r")
    readerGallery = csv.reader(galleryFile)
    for item in readerGallery:
        img1_path = img_dir + item[1]
        img1 = Image.open(img1_path).convert("RGB")
        img1 = data_transforms(img1)
        with torch.no_grad():
            img1 = Variable(img1.unsqueeze(0)).cuda()
        galleryfeature = net(img1,img1,img1)[0]
        galleryFeature.append(galleryfeature.data.cpu().numpy())
    galleryFile.close()

    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    metric = cosmetric(galleryFeature, probeFeature)

    k = 0
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = csv.reader(csvFile)

    for item in readerC:
        if metric[int(item[0])] == int(item[1]):
            k += 1
    auc = k / len(metric)
    print('accuracy:{}'.format(auc))
    return acc

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
import argparse
parser = argparse.ArgumentParser(description='Triplet')
parser.add_argument('--resume', dest='resume',help='resume', type=str, default=None)
parser.add_argument('--metric', dest='metric',help='metric', type=str, default='softmax')
parser.add_argument('--margin', dest='margin',help='the distance of positive and negative', type=float, default=0.35)
parser.add_argument('--epoch',dest='epoch',type=int,default=50)
parser.add_argument('--batch_size',dest='batch_size',type=int,default=32)
parser.add_argument('--train-dataset', dest='train_dataset',
                        help='train-dataset', type=str,default="../af2019-ksyun-training-20190416/")
parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str,default="../finalA_crop_dataset_enlarge/")
parser.add_argument('--model', dest='model',help='save model dir', type=str, default='triplet_models')
args = parser.parse_args()

data_dir = args.train_dataset
test_dir = args.test_dataset

if not os.path.exists(args.model):
    os.mkdir(args.model)
root = '/home/xuanni/codework/workdir/aifuture/crop_dataset_enlarge_train'
bs = args.batch_size

triplet_datasets = TripletFace(root,data_transforms['train'])
num_class = triplet_datasets.num_class
net = TripletNetwork(num_class)
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net = net.cuda()
start_epoch = 0

if args.resume:
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint)
    print('load model successfully!')
    start_epoch = int(args.resume.split('_')[-1][:-4]) + 1
for param in net.parameters():
    param.requires_grad = True

margin = args.margin

loss_fn = nn.CrossEntropyLoss()

criterion = nn.TripletMarginLoss(margin=margin,p=1)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = adabound.AdaBound(net.parameters(), lr=1e-3, final_lr=0.1)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = args.epoch
best_acc = 0
for epoch in range(start_epoch, epochs):
    triplet_datasets = TripletFace(root,data_transforms['train'])
    train_loaders =torch.utils.data.DataLoader(triplet_datasets,
                                                  batch_size=bs,
                                                  shuffle=True,
                                                  num_workers=16,pin_memory=True)
    model = train(net,criterion, loss_fn, optimizer, epoch,scheduler=exp_lr_scheduler)
    acc = test(model,test_dir) 
    if acc>best_acc:
        torch.save(model.state_dict(), os.path.join(args.model, args.metric+'_triple_'+str(epoch)+'.pth'))
        print('saved model of epoch:{}'.format(epoch))
        best_acc = acc
