from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'common'))
import face_image
import face_preprocess

class FaceModel(object):
  def __init__(self, detector = None, model = None):
    self.TEST_SCALES = [640]
    self.target_size = (640, 640)
    self.detector = detector
    self.model = model

  def get_scales(self, img):
    im_shape = img.shape
    target_size = self.target_size[0]
    max_size = self.target_size[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [float(scale)/target_size*im_scale for scale in self.TEST_SCALES]
    return scales
  
  def get_annots(self, img, **kwargs):
    scales = self.get_scales(img)
    ret = self.detector.detect(img, scales=scales, **kwargs)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0, 0:4]
    return bbox, points

  def get_input(self, img, align=False,**kwargs):
    res = self.get_annots(img, **kwargs)
    if res is None:
      return None
    bbox, points = res

    if align:
      nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
    else:
      nimg = face_preprocess.preprocess(img, bbox, margin=30)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    return nimg

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding
