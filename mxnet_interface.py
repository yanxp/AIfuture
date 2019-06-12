# -*- coding:utf-8 -*-
from __future__ import print_function, division
import os
import cv2
import numpy as np
import scipy.spatial.distance as spd
import mxnet as mx
from face_model import FaceModel
from retinaface import RetinaFace

def detect_or_return_origin(img_path, model, align=False):
    # return RGB image (c, h, w)
    img = cv2.imread(img_path)
    new_img = model.get_input(img, threshold=0.02, align=align)

    if new_img is None:
        img = cv2.resize(img, (142, 142))
        b = (142 - 112) // 2
        img = img[b:-b, b:-b, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(img, (2,0,1))
        return aligned
    else:
        new_img = cv2.resize(new_img, (112, 112))
        aligned = np.transpose(new_img, (2,0,1))
        return aligned

def cal_metric(galleryFeature, probeFeature, dis_type="cosine", THRESHOD = 0.3, contains_flip = False):
    LEN_THRESHOD = max(1, int(len(galleryFeature) * 0.25)) # 1 <= x <= 10
    res = []
    metricMat = spd.cdist(probeFeature, galleryFeature, dis_type)
    for i, metric in enumerate(metricMat):
        idx = np.argsort(metric)
        if metric[idx[LEN_THRESHOD]] - metric[idx[0]] >= THRESHOD:
            if contains_flip:
                res.append(idx[0]//2)
            else:
                res.append(idx[0])
        else:
            res.append(-1)
    return res

def predict_interface(imgset_rpath: str, gallery_dict: dict, probe_dict: dict) -> [(str, str), ...]:
    align_match = bool(os.getenv('ALIGN_MATCH'))
    # 1. load model
    if align_match:
        detector = RetinaFace("./models/R50", 0, 0, 'net3', 0.4, False, vote=False)
        _, arg_params, aux_params = mx.model.load_checkpoint('./models/testR50', 4)
        detector.model.set_params(arg_params, aux_params, allow_missing = True)
    else:
        detector = RetinaFace("./models/testR50", 4, 0, 'net3', 0.4, False, vote=False)
        
    path, epoch = os.getenv('PRETRAINED_MODEL').split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, int(epoch))
    model = mx.mod.Module(context = mx.gpu(0), symbol = sym)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    fmodel = FaceModel(detector, model)
    # 2. get features
    probe_list = [(k, v) for k, v in probe_dict.items()]
    gallery_list = [(k, v) for k, v in gallery_dict.items()]
    galleryFeature = []
    probeFeature = []
    prob_imgs = []
    gallery_imgs = []
    for _, item in probe_list:
        img0_path = os.path.join(imgset_rpath, item)
        img0 = detect_or_return_origin(img0_path, fmodel, align=align_match)
        prob_imgs.append(img0)

    for _, item in gallery_list:
        img1_path = os.path.join(imgset_rpath, item)
        img1 = detect_or_return_origin(img1_path, fmodel, align=align_match)
        gallery_imgs.append(img1)
    # 3. face recogonition
    for img0 in prob_imgs:
        probefeature = fmodel.get_feature([img0])
        probeFeature.append(probefeature)
    for img1 in gallery_imgs:
        galleryfeature = fmodel.get_feature([img1])
        galleryFeature.append(galleryfeature)
    
    galleryFeature = mx.ndarray.concat(*galleryFeature, dim=0).asnumpy()
    probeFeature = mx.ndarray.concat(*probeFeature, dim=0).asnumpy()
    preds = cal_metric(galleryFeature, probeFeature, "cosine", 0.15)

    result = [] # result = [("1", "2"), ("2", "4")]
    for i, p in enumerate(preds):
        if p != -1:
            result.append((probe_list[i][0], gallery_list[p][0]))
        else:
            result.append((probe_list[i][0], "-1"))
    return result
