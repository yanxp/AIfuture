# -*- coding:utf-8 -*-
from __future__ import print_function, division
import os
import cv2
import numpy as np
import scipy.spatial.distance as spd

import multiprocessing

def init():
    global mx
    global fmodel
    global detector
    global model
    global align_match
    import mxnet as mx
    from face_model import FaceModel
    from retinaface import RetinaFace
    align_match = bool(os.getenv('ALIGN_MATCH'))
    if align_match:
        detector = RetinaFace("./models/finalR50", 0, 0, 'net3', 0.4, False, vote=False)
    else:
        detector = RetinaFace("./models/testR50", 4, 0, 'net3', 0.4, False, vote=False)
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, int(epoch))
    model = mx.mod.Module(context = mx.gpu(0), symbol = sym)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    fmodel = FaceModel(detector, model)

def detect_or_return_origin(img_path):
    global fmodel
    global align_match
    # return RGB image (c, h, w)
    img = cv2.imread(img_path)
    new_img = fmodel.get_input(img, threshold=0.02, align=align_match)

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

def get_batch_feature(img_batch):
    global fmodel
    return fmodel.get_feature(img_batch)

def get_res_from_metric(metric_row):
    THRESHOD = 0.15
    idx = np.argsort(metric_row)
    if metric_row[idx[LEN_THRESHOD]] - metric_row[idx[0]] >= THRESHOD:
        return idx[0]
    else:
        return -1

def predict_interface(imgset_rpath: str, gallery_dict: dict, probe_dict: dict) -> [(str, str), ...]:
    pool = multiprocessing.Pool(4, initializer = init)
    
    probe_list = [(k, v) for k, v in probe_dict.items()]
    prob_imgs = pool.map_async(detect_or_return_origin, 
        (os.path.join(imgset_rpath, v) for k, v in probe_list))
    gallery_list = [(k, v) for k, v in gallery_dict.items()]
    gallery_imgs = pool.map_async(detect_or_return_origin, 
        (os.path.join(imgset_rpath, v) for k, v in gallery_list))
    
    pool2 = multiprocessing.Pool() # hide overhead ?

    batch_size = int(os.getenv('BATCH_SIZE'))
    gallery_imgs = gallery_imgs.get()
    galleryFeature = pool.map_async(get_batch_feature, 
        (gallery_imgs[i:i + batch_size] for i in range(0, len(gallery_imgs), batch_size) ))
    prob_imgs = prob_imgs.get()
    probeFeature = pool.map_async(get_batch_feature, 
        (prob_imgs[i:i + batch_size] for i in range(0, len(prob_imgs), batch_size) ))
    
    galleryFeature = galleryFeature.get()
    galleryFeature = mx.ndarray.concat(*galleryFeature, dim=0).asnumpy()
    probeFeature = probeFeature.get()
    probeFeature = mx.ndarray.concat(*probeFeature, dim=0).asnumpy()
    # calculate cosine distance
    LEN_THRESHOD = max(1, int(len(galleryFeature) * 0.25))
    metricMat = spd.cdist(probeFeature, galleryFeature, "cosine")

    preds = pool2.map(get_res_from_metric, metricMat)
    
    result = [] # result = [("1", "2"), ("2", "4")]
    for i, p in enumerate(preds):
        if p != -1:
            result.append((probe_list[i][0], gallery_list[p][0]))
        else:
            result.append((probe_list[i][0], "-1"))
    return result