# %matplotlib inline
from __future__ import print_function, division
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from PIL import Image
import argparse
import torchfile
import os
import pickle
import random

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def parse_args():
    parser = argparse.ArgumentParser(description='Network')
    parser.add_argument('--resume', dest='resume', help='resume model', type=str, default=None)
    parser.add_argument('--training-dataset', dest='train_dataset',
                        help='train-dataset', type=str, default="af2019-ksyun-training-20190416/")
    parser.add_argument('--epochs', dest='epochs', help='numbers of epoch', type=int, default=30)
    parser.add_argument('--bs', dest='bs', help='batch size', type=int, default=64)
    parser.add_argument('--model', dest='model', help='diretory of models', type=str, default="./")
    args = parser.parse_args()
    return args

def getExample (path , n):
    example_list = []
    return_list = []
    for example in os.listdir(path):
        examplePath = os.path.join(path,example)
        example_list.append(examplePath)
    return_list = list(random.sample(example_list, n))
    return return_list

def getClass (path):
    classPath = path.split('/')[-3]
    return classPath

def makedata(proot,root):
    if os.path.exists(proot + 'data_train.pkl'):
        return
    trainList = []
    pExamleList  = []
    cExamleList = []
    thresholdNum = 6
    for person in os.listdir(root):
        if os.path.isfile(os.path.join(root, person)): break
        count = 0
        person_dir = os.path.join(root, person, '1')
        if os.path.isfile(person_dir): break
        for pictures in os.listdir(person_dir):
            count += 1
        if count >= thresholdNum:
            photo = getExample(person_dir, thresholdNum)
        else:
            photo = getExample(person_dir, count)
        pExamleList.append(photo)

        count = 0
        person_dir = os.path.join(root, person, '0')
        for pictures in os.listdir(person_dir):
            count += 1
        if count >= thresholdNum:
            caricature = getExample(person_dir, thresholdNum)
        else:
            caricature = getExample(person_dir, count)
        cExamleList.append(caricature)

    number = len(pExamleList)
    for i in range(number):
        pList = pExamleList[i]
        cList = cExamleList[i]
        pClassName = getClass(pList[0])
        flag = True
        while flag:
            cClass = getExample(root, thresholdNum)
            for cClassName in cClass:
                cClassName = cClassName.split('/')[-1]
                if pClassName == cClassName:
                    flag = True
                    break
                else:
                    flag = False

        for p in pList:
            for q in cList:
                for c in cClass:
                    c = c+'/0'
                    c = getExample(c,1)
                    one = (p,q,c[0])
                    trainList.append(one)
    save_path = proot + 'data_train.pkl'
    output = open(save_path, 'wb')
    print(save_path)
    pickle.dump(trainList, output)
    output.close()

def make_dataset(dir):
    data_root = os.path.join(dir, 'data_train.pkl')
    images = pickle.load(open(data_root, 'rb'))
    return images


def train(model, criterion, optimizer, epoch, scheduler=None):
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
        dist1 = F.pairwise_distance(outputs[0], outputs[1])
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
            print("Epoch number {}; batch_idx:{} ;Current loss {}".format(epoch, batch_idx, np.mean(losses)))

    total_loss /= (batch_idx + 1)
    print('total_loss:', total_loss)
    print('positive_distance :', positive_distance / (batch_idx + 1))
    print('negative_distance :', negative_distance / (batch_idx + 1))
    return model


block_size = [2, 2, 3, 3, 3]


def load_weights(net, path="VGG_FACE.t7"):
    model = torchfile.load(path)
    counter = 1
    block = 1
    k = 0

    for i, layer in enumerate(model.modules):
        self_layer = None
        if layer.weight is not None:
            if block <= 5:
                while self_layer is None:
                    if isinstance(net[k], nn.Conv2d):
                        self_layer = net[k]
                        k += 1
                    else:
                        k += 1
                counter += 1
                if counter > block_size[block - 1]:
                    counter = 1
                    block += 1
                self_layer.weight.data[...] = torch.Tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.Tensor(layer.bias).view_as(self_layer.bias)[...]
            else:

                while self_layer is None:
                    if not isinstance(net[k], Lambda) and isinstance(net[k], nn.Sequential):
                        self_layer = net[k][1]
                        k += 1
                    else:
                        k += 1
                block += 1
                self_layer.weight.data[...] = torch.Tensor(layer.weight).view_as(self_layer.weight)[...]
                self_layer.bias.data[...] = torch.Tensor(layer.bias).view_as(self_layer.bias)[...]


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


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.cnn = VGG_FACE
        load_weights(self.cnn, 'VGG_FACE.t7')
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


class NetLoss(nn.Module):
    def __init__(self, margin):
        super(NetLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class FaceDataset(Dataset):
    def __init__(self, img_path, transform=None):
        # class_mapping = {cls: i for i, cls in enumerate(os.listdir(os.path.join(img_path, 'images')))}
        class_mapping = {cls: i for i, cls in enumerate(os.listdir(img_path))}
        self.labels = []
        self.sets = make_dataset(img_path)
        for data in self.sets:
            self.labels.append(class_mapping[data[0].split('/')[-3]])
        self.transforms = transform

    def __getitem__(self, index):
        anchor, positive, negative = self.sets[index]
        anchor_img = Image.open(anchor).convert('RGB')
        positive_img = Image.open(positive).convert('RGB')
        negative_img = Image.open(negative).convert('RGB')
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)
        return (anchor_img, positive_img, negative_img), self.labels[index]

    def __len__(self):
        return len(self.sets)

if __name__ == '__main__':
    args = parse_args()
    data_dir = args.train_dataset
    bs = args.bs

    if not os.path.exists(args.model):
        os.makedir(args.model)
    if data_dir[-1] is not '/':
        data_dir = data_dir + '/'
    # root = data_dir + "images/"
    root = data_dir 

    makedata(data_dir,root)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    datasets = {'train':FaceDataset(data_dir,
                               data_transforms['train'])}

    train_loaders = torch.utils.data.DataLoader(datasets['train'],
                                                batch_size=bs,
                                                shuffle=True,
                                                num_workers=16,
                                                pin_memory=True)

    net = Network()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = net.cuda()
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint)
        print('load model successfully!')
        start_epoch = int(args.resume[7:-4]) + 1
    for param in net.parameters():
        param.requires_grad = True

    margin = 1.0
    criterion = NetLoss(margin)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = args.epochs
    count_history = []
    loss_history = []
    auc_histoty = []

    for epoch in range(start_epoch, epochs):
        count_history.append(epoch)
        model = train(net, criterion, optimizer, epoch, scheduler=exp_lr_scheduler)
        torch.save(model.state_dict(), os.path.join(args.model, 'Network' + str(epoch) + '.pth'))
