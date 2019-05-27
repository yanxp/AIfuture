# -*- coding:utf-8 -*-
from __future__ import print_function, division
import time
import argparse
import os
from PIL import Image
import cv2
import numpy as np
from sklearn import metrics

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms

import VGG_FACE
from deploy import face_model
class FaceModelArgs(object):
    def __init__(self):
        self.image_size = "256,256"
        self.gpu = 0
        self.det = 0
        self.flip = 0
        self.threshold = 1.24
        self.model = ''

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn = VGG_FACE.VGG_FACE
        module_list = list(self.cnn.children())
        self.model = nn.Sequential(*module_list[:-4])
    def forward(self, x):
        out = self.model(x)
        return out

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

        metric.append((d["index"], d['value']))
    return metric

def edumetric(galleryFeature, probeFeature, THRESHOD = 0.4):
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

def detect_or_return_origin(img_path, model):
    img = cv2.imread(img_path)
    new_img = model.get_input(img)
    if new_img is None:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(new_img)

def predict_interface(imgset_rpath: str, gallery_dict: dict, probe_dict: dict) -> [(str, str), ...]:
    """
    imgset_rpath: 数据图片库的相对路径，例如数据集的相对路径为test_set/testB，那么图片库的相对路径为test_set/testB/images
    在run_test.sh中已经设置了数据集的默认地址

    gallery_dict:  {gallery_pic_id: pic_rpath,...}，类型为dict。
    其中gallery_pic_id为图片在gallery库中的id，类型为str，下同；
    pic_rpath 为图片在图片库中的相对路径，类型为str，例如test_set/testB/images/dd/dd7e12af086264e1b13d8a788857077d.jpg

    probe_dict:    {probe_pic_id: pic_rpath,...}，类型为dict。
    其中probe_pic_id为图片在probe库中的id，类型为str，下同；
    pic_rpath 为图片在图片库中的相对路径，类型为str，例如test_set/testB/images/65/65125c57356c12c916b00afcd572b1ec.jpg

    :return [(probe_pic_id, gallery_pic_id),...]，类型为list。
    其中元素是由probe_pic_id和gallery_pic_id组成的tuple。
    """
    # 1. load model and other settings
    data_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    net = TripletNetwork()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    load_name = os.getenv('PRETRAINED_MODEL')
    checkpoint = torch.load(load_name)
    net.load_state_dict(checkpoint)
    net = net.cuda()
    net.eval()

    detector = face_model.FaceModel(FaceModelArgs())
    # 2. get features
    probe_list = [(k, v) for k, v in probe_dict.items()]
    gallery_list = [(k, v) for k, v in gallery_dict.items()]
    galleryFeature = []
    probeFeature = []
    prob_imgs = []
    gallery_imgs = []
    for _, item in probe_list:
        img0_path = os.path.join(imgset_rpath, item)
        img0 = detect_or_return_origin(img0_path, detector)
        prob_imgs.append(img0)

    for _, item in gallery_list:
        img1_path = os.path.join(imgset_rpath, item)
        img1 = detect_or_return_origin(img1_path, detector)
        gallery_imgs.append(img1)
    del detector

    for img0 in prob_imgs:
        #img0 = Image.open(img0_path).convert("RGB")
        img0 = data_transforms(img0)
        img0 = Variable(img0.unsqueeze(0), volatile=True).cuda()
        probefeature = net(img0)
        probeFeature.append(probefeature.data.cpu().numpy())

    for img1 in gallery_imgs:
        #img1 = Image.open(img1_path).convert("RGB")
        img1 = data_transforms(img1)
        img1 = Variable(img1.unsqueeze(0), volatile=True).cuda()
        galleryfeature = net(img1)
        galleryFeature.append(galleryfeature.data.cpu().numpy())

    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    preds = edumetric(galleryFeature, probeFeature)

    # 3. prepare result
    result = [] # result = [("1", "2"), ("2", "4")]
    for i, p in enumerate(preds):
        if p != -1:
            result.append((probe_list[i][0], gallery_list[p][0]))
        else:
            result.append((probe_list[i][0], "-1"))

    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description='Welcome to ACG Face contest! Please input your argument according to usage.'
    )
    parser.add_argument(
        '--data_rpath',
        dest='data_rpath',
        help='relative path of dataset',
        type=str
    )
    parser.add_argument(
        '--team_number',
        dest='team_number',
        help='the number of team',
        type=str
    )
    parser.add_argument(
        '--version_number',
        dest='version_number',
        help='version number of work submitted',
        type=str
    )
    args = parser.parse_args()
    return args


def run_test():
    args = parse_args()
    data_rpath = args.data_rpath
    team_number = args.team_number
    version_number = args.version_number

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

    t_start = time.time()

    result_list = predict_interface(
        imgset_rpath=os.path.join(data_rpath, 'images'),
        gallery_dict=gallery_dict,
        probe_dict=probe_dict
    )

    t_end = time.time()
    duration = t_end - t_start

    result_str = 'Team_number: {}, version_number: {}, total_time: {:.1f}seconds.\n'\
        .format(team_number, version_number, duration)

    for result_i in result_list:
        result_str += '{}, {}\n'.format(result_i[0], result_i[1])

    with open('result_%s_%s.csv' % (team_number, version_number), 'w') as output_file:
        output_file.write(result_str)


if __name__ == '__main__':
    run_test()
