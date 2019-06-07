# %matplotlib inline
from __future__ import print_function, division
import argparse
import csv
import numpy as np
import os
from PIL import Image
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms

from face_model import FaceModel
from retinaface import RetinaFace
import pytorch_interface

def parse_args():
    parser = argparse.ArgumentParser(description='Test Face Recogonition task')
    parser.add_argument('--data_rpath', dest='data_rpath', help='relative path of dataset')
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
    # 1. set path
    data_rpath = args.data_rpath
    gallery_dict = {}
    probe_dict = {}
    probe_csv = os.path.join(data_rpath, 'probe.csv')
    for line in open(probe_csv):
        probe_pic_id = line.strip().split(',')[0]
        probe_pic_url = line.strip().split(',')[1]
        probe_dict[probe_pic_id] = probe_pic_url

    gallery_csv = os.path.join(data_rpath, 'gallery.csv')
    for line in open(gallery_csv):
        gallery_pic_id = line.strip().split(',')[0]
        gallery_pic_url = line.strip().split(',')[1]
        gallery_dict[gallery_pic_id] = gallery_pic_url
    imgset_rpath = os.path.join(data_rpath, 'images')
    if args.vis:
        vis_probe_1 = 'vis/probe/1'
        os.makedirs(vis_probe_1, exist_ok=True)
        vis_probe_0 = 'vis/probe/0'
        os.makedirs(vis_probe_0, exist_ok=True)
        vis_gallery_1 = 'vis/gallery/1'
        os.makedirs(vis_gallery_1, exist_ok=True)
        vis_gallery_0 = 'vis/gallery/0'
        os.makedirs(vis_gallery_0, exist_ok=True)
    # -------------------------------
    # 2. model load
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    net = pytorch_interface.TripletNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    checkpoint = torch.load(args.model)
    checkpoint = {k: checkpoint[k] for k in net.state_dict().keys() }
    net.load_state_dict(checkpoint)
    net = net.cuda()
    net.eval()

    detector = RetinaFace(args.pdetect, args.depoch, 0, args.dnet, args.nms, args.nocrop, vote=False)
    fmodel = FaceModel(detector)
    # -------------------------------
    # 3. get feature
    probe_list = [(k, v) for k, v in probe_dict.items()]
    gallery_list = [(k, v) for k, v in gallery_dict.items()]
    galleryFeature = []
    probeFeature = []
    prob_imgs = []
    gallery_imgs = []

    for _, item in probe_list:
        img0_path = os.path.join(imgset_rpath, item)
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

    for _, item in gallery_list:
        img1_path = os.path.join(imgset_rpath, item)
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
        metric = pytorch_interface.edumetric(galleryFeature, probeFeature, th)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
        auc = k / len(metric)
        print('threshod: {} , correct: {}, auc: {}'.format(th, k, auc))
