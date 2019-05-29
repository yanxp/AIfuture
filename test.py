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
from retinaface import RetinaFace
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str)
    parser.add_argument('--model', dest='model',
                        help='model', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    # RetinaNet: prefix, epoch, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4 = 0.5, vote=False
    parser.add_argument('--pretrained-detector', dest="pdetect",
                        help="detector checkpoint prefix", default="./models/R50")
    parser.add_argument('--detector-epoch', dest='depoch', default=0)
    parser.add_argument('--detector-network', dest="dnet",
                        help="detector config type", default='net3')
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--nocrop', action="store_true")
    parser.add_argument('--box-vote', action="store_true", dest="dVote")
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

def get_image_scales(img):
    scales = [320, 640] # min size, max size
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return [im_scale]

def detect_or_return_origin(img_path, model):
    img = cv2.imread(img_path)
    scales = get_image_scales(img)
    faces, landmarks = model.detect(img, scales=scales)

    if faces is None:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        det = faces[0].astype(np.int)
        margin = 44 # extend the box
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        new_img = img[bb[1]:bb[3],bb[0]:bb[2],:] 
        return Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

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
    prob_imgs = []
    gallery_imgs = []

    detector = RetinaFace(args.pdetect, args.depoch, 0, args.dnet, args.nms, args.nocrop, vote=args.dVote)

    probeFile = open(probe, "r")
    readerProbe = csv.reader(probeFile)
    for _, item in readerProbe:
        img0_path = os.path.join(img_dir, item)
        img0 = detect_or_return_origin(img0_path, detector)
        prob_imgs.append(img0)
    probeFile.close()

    galleryFile = open(gallery, "r")
    readerGallery = csv.reader(galleryFile)
    for _, item in readerGallery:
        img1_path = os.path.join(img_dir, item)
        img1 = detect_or_return_origin(img1_path, detector)
        gallery_imgs.append(img1)
    galleryFile.close()

    del detector

    for img0 in prob_imgs:
        img0 = data_transforms(img0)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        probefeature = net(img0)
        probeFeature.append(probefeature.data.cpu().numpy())


    for img1 in gallery_imgs:
        img1 = data_transforms(img1)
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        galleryfeature = net(img1)
        galleryFeature.append(galleryfeature.data.cpu().numpy())

    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = list(csv.reader(csvFile))

    for th in [0.375, 0.4, 0.425, 0.45]:
        k = 0
        metric = edumetric(galleryFeature, probeFeature, th)
        #metric = cosmetric(galleryFeature, probeFeature)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
        auc = k / len(metric)
        print('threshod: {} , correct: {}, auc: {}'.format(th, k, auc))
