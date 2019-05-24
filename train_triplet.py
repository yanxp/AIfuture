# %matplotlib inline
from __future__ import print_function, division

import torchfile as torchfile
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
from dataloader import TripletFace
import VGG_FACE
from torch.optim import lr_scheduler
import math

from VGG_FACE import load_weights

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        load_weights(self.cnn,'VGG_FACE.t7')
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2,input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

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

def train(model, criterion, optimizer, epoch,scheduler=None):
    losses = []
    total_loss = 0
    if scheduler is not None:
        scheduler.step()
        model.train(True)  # Set model to training mode
    else:
        model.train(True)  # Set model to training mode
    positive_distance = 0
    negative_distance = 0
    for batch_idx, (data, target) in enumerate(train_loaders):
        if not type(data) in (tuple, list):
            data = (data,)
        data = tuple(Variable(d.cuda()) for d in data)
        optimizer.zero_grad()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        dist1 = F.pairwise_distance(outputs[0],outputs[1])
        positive_distance += np.mean(dist1.cpu().data.numpy())
        dist2 = F.pairwise_distance(outputs[0], outputs[2])
        negative_distance += np.mean(dist2.cpu().data.numpy())
        loss_inputs = outputs
        loss_outputs = criterion(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.data[0])
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print("Epoch number {}; batch_idx:{} ;Current loss {}".format(epoch,batch_idx,np.mean(losses)))

    total_loss /= (batch_idx + 1)
    print('total_loss:',total_loss)
    print('positive_distance :',positive_distance/(batch_idx + 1))
    print('negative_distance :', negative_distance / (batch_idx + 1))
    return model

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
import argparse
parser = argparse.ArgumentParser(description='Triplet')
parser.add_argument('--resume', dest='resume',help='resume', type=str, default=None)
parser.add_argument('--training-dataset', dest='train_dataset',help='train-dataset', type=str,default="../af2019-ksyun-training-20190416/")
parser.add_argument('--model', dest='model',help='save model dir', type=str, default='triplet_models')
args = parser.parse_args()

data_dir = args.train_dataset
if not os.path.exists(args.model):
    os.mkdir(args.model)

triplet_datasets = {x: TripletFace(data_dir,x,
                                  data_transforms[x])
                  for x in ['train', 'val']}
bs = 32
train_loaders =torch.utils.data.DataLoader(triplet_datasets['train'],
                                              batch_size=bs,
                                              shuffle=True,
                                              num_workers=16,
                                              pin_memory=True)
net = TripletNetwork()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net = net.cuda()
start_epoch = 0
if args.resume:
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint)
    print('load model successfully!')
    start_epoch = int(args.resume[14:-4]) + 1
for param in net.parameters():
    param.requires_grad = True

margin = 1.0
criterion = TripletLoss(margin)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 50
count_history = []
loss_history = []
auc_histoty = []

for epoch in range(start_epoch, epochs):
    count_history.append(epoch)
    model = train(net, criterion, optimizer, epoch,scheduler=exp_lr_scheduler)
    torch.save(model.state_dict(), os.path.join(args.model,'tripletnetwork'+str(epoch)+'.pth'))
