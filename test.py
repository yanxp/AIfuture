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
import VGG_FACE
import cv2
import torch.nn.functional as F
from face_model import FaceModel
from retinaface import RetinaFace

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str)
    parser.add_argument('--model', dest='model',
                        help='model', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--save_crop', action="store_true")
    parser.add_argument('--crop_dir', default="../rf-finalA-cropped")
    # RetinaNet: prefix, epoch, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4 = 0.5, vote=False
    parser.add_argument('--pretrained-detector', dest="pdetect",
                        help="detector checkpoint prefix", default="./models/testR50")
    parser.add_argument('--detector-epoch', dest='depoch', default=4, type=int)
    parser.add_argument('--detector-network', dest="dnet",
                        help="detector config type", default='net3')
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--nocrop', action="store_true")
    args = parser.parse_args()
    return args

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
        out = F.normalize(out, p=2, dim=1)
        return out

def detect_or_return_origin(img_path, model):
    img = cv2.imread(img_path)
    new_img = model.get_input(img, threshold=0.02)

    if new_img is None:
        img = cv2.resize(img, (256, 256))
        b = (256 - 224) // 2
        img = img[b:-b, b:-b, :]
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), False
    else:
        new_img = cv2.resize(new_img, (224, 224))
        return Image.fromarray(new_img), True

if __name__ == '__main__':
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = TripletNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    checkpoint = torch.load(args.model)
    checkpoint = {k: checkpoint[k] for k in net.state_dict().keys() }
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
    if args.vis:
        vis_probe_1 = 'vis/probe/1'
        os.makedirs(vis_probe_1, exist_ok=True)
        vis_probe_0 = 'vis/probe/0'
        os.makedirs(vis_probe_0, exist_ok=True)
        vis_gallery_1 = 'vis/gallery/1'
        os.makedirs(vis_gallery_1, exist_ok=True)
        vis_gallery_0 = 'vis/gallery/0'
        os.makedirs(vis_gallery_0, exist_ok=True)

    detector = RetinaFace(args.pdetect, args.depoch, 0, args.dnet, args.nms, args.nocrop, vote=False)
    fmodel = FaceModel(detector)

    probeFile = open(probe, "r")
    readerProbe = csv.reader(probeFile)
    for _, item in readerProbe:
        img0_path = os.path.join(img_dir, item)
        img0, hit = detect_or_return_origin(img0_path, fmodel)
        prob_imgs.append(img0)
        if args.vis:
            if hit:
                img0.save(os.path.join(vis_probe_1, item[-10:]))
            else:
                img0.save(os.path.join(vis_probe_0, item[-10:]))
        if args.save_crop:
            os.makedirs(os.path.join(args.crop_dir,'images',os.path.dirname(item)), exist_ok=True)
            img0.save(os.path.join(args.crop_dir, 'images', item))
    probeFile.close()

    galleryFile = open(gallery, "r")
    readerGallery = csv.reader(galleryFile)
    for _, item in readerGallery:
        img1_path = os.path.join(img_dir, item)
        img1, hit = detect_or_return_origin(img1_path, fmodel)
        gallery_imgs.append(img1)
        if args.vis:
            if hit:
                img1.save(os.path.join(vis_gallery_1, item[-10:]))
            else:
                img1.save(os.path.join(vis_gallery_0, item[-10:]))
        if args.save_crop:
            os.makedirs(os.path.join(args.crop_dir,'images',os.path.dirname(item)), exist_ok=True)
            img1.save(os.path.join(args.crop_dir, 'images', item))
    galleryFile.close()

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
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = list(csv.reader(csvFile))

    for th in np.arange(0.1,0.3,0.02): #0.166
        k = 0
        metric = edumetric(galleryFeature, probeFeature, th)
        #metric = cosmetric(galleryFeature, probeFeature)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
        auc = k / len(metric)
        print('threshod: {} , correct: {}, auc: {}'.format(th, k, auc))
