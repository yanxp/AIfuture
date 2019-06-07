from __future__ import print_function, division
import argparse
import csv
import numpy as np
import os
import mxnet as mx
from retinaface import RetinaFace
from face_model import FaceModel
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Test Face Recogonition task')
    parser.add_argument('--test-dataset', dest='test_dataset',
                        help='test-dataset', type=str)
    parser.add_argument('--model', dest='model',
                        help='model ckpt path', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--save_crop', action="store_true")
    parser.add_argument('--crop_dir', default="../rf-finalA-cropped")
    parser.add_argument('--type', default='l1', choices=['l1','l2','cos'])
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

def cal_metric(galleryFeature, probeFeature, dis_type="l2", THRESHOD = 0.3):
    LEN_THRESHOD = max(1, int(len(galleryFeature) * 0.25)) # 1 <= x <= 10
    res = []
    for i, p in enumerate(probeFeature):
        metric = np.zeros( (len(galleryFeature),) )
        # p = p / np.linalg.norm(p)
        for j, g in enumerate(galleryFeature):
            # g = g / np.linalg.norm(g)
            if dis_type == "l2":
                metric[j] = np.sum((p - g) ** 2)
            elif dis_type == "l1":
                metric[j] = np.sqrt(np.sum((p - g)**2))
            elif dis_type == 'cos':
                metric[j] = - np.sum(p * g) # from large to small
        
        idx = np.argsort(metric)
        if metric[idx[LEN_THRESHOD]] - metric[idx[0]] >= THRESHOD:
            res.append(idx[0])
        else:
            res.append(-1)
    return res
    
def detect_or_return_origin(img_path, model):
    # return RGB image (c, h, w)
    img = cv2.imread(img_path)

    # remenber to delete when in interface.py!
    if img.shape[0] == 224 and img.shape[1] == 224:
        img = cv2.resize(img, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(img, (2,0,1))
        return aligned, True

    new_img = model.get_input(img, threshold=0.02)

    if new_img is None:
        img = cv2.resize(img, (256, 256))
        b = (256 - 224) // 2
        img = img[b:-b, b:-b, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(img, (2,0,1))
        return aligned, False
    else:
        new_img = cv2.resize(new_img, (224, 224))
        aligned = np.transpose(new_img, (2,0,1))
        return aligned, True

if __name__ == '__main__':
    args = parse_args()
    # 1. paths setting
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
    # -------------------------
    fmodel = FaceModel()
    # 2. face detection
    path, epoch = args.model.split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, int(epoch))
    model = mx.mod.Module(context = mx.gpu(0), symbol = sym)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    fmodel.model = model
    
    detector = RetinaFace(args.pdetect, args.depoch, 0, args.dnet, args.nms, args.nocrop, vote=False)
    fmodel.detector = detector
    with open(probe, "r") as probeFile:
        readerProbe = csv.reader(probeFile)
        for _, item in readerProbe:
            img0_path = os.path.join(img_dir, item)
            img0, hit = detect_or_return_origin(img0_path, fmodel)
            prob_imgs.append(img0)

    with open(gallery, "r") as galleryFile:
        readerGallery = csv.reader(galleryFile)
        for _, item in readerGallery:
            img1_path = os.path.join(img_dir, item)
            img1, hit = detect_or_return_origin(img1_path, fmodel)
            gallery_imgs.append(img1)
    # -------------------------
    # 3. face recogonition
    for img0 in prob_imgs:
        probefeature = fmodel.get_feature(img0)
        probeFeature.append(probefeature)

    for img1 in gallery_imgs:
        galleryfeature = fmodel.get_feature(img1)
        galleryFeature.append(galleryfeature)
    # -------------------------
    # 4. prediction
    galleryFeature = np.array(galleryFeature)
    probeFeature = np.array(probeFeature)
    filename = rootpath + "ground_truth.csv"
    csvFile = open(filename, 'r')
    readerC = list(csv.reader(csvFile))

    for th in np.arange(0, 1, 0.1):
        k = 0
        metric = cal_metric(galleryFeature, probeFeature, args.type, th)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
        auc = k / len(metric)
        print('threshod: {} , correct: {}, auc: {}'.format(th, k, auc))
