from __future__ import print_function

import argparse
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
from retinaface import RetinaFace


def parse_args():
    parser = argparse.ArgumentParser(description='Test widerface by retinaface detector')
    # general
    parser.add_argument('--network', help='network name', default='net3', type=str)
    parser.add_argument('--data_rpath', help='dataset name', default='retinaface', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default='', type=str)
    parser.add_argument('--epoch', help='model to test with', default=0, type=int)
    parser.add_argument('--output', help='output folder', default='rf-cropped-training', type=str)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--nocrop', action="store_true")
    args = parser.parse_args()
    return args

detector = None
args = None
all_img_num = 0
not_detected = 0

def test(args):
  print('test with', args)
  global detector
  output_root = os.path.join(args.output, 'images')
  if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
  detector = RetinaFace(args.prefix, args.epoch, 0, args.network, args.nms, nocrop=args.nocrop, vote=True)

  for cls in os.listdir(args.data_rpath):
    for domain in ['0', '1']:
      path = os.path.join(args.data_rpath, cls, domain)
      for imgn in os.listdir(path):
        imgp = os.path.join(path, imgn)
        img = cv2.imread(imgp)
        new_img = detector.get_input(img, threshold=0.02)
        if new_img is None:
          not_detected += 1
          img.imwrite(os.path.join(output_root, cls, domain, imgn))
        else:
          new_img.imwrite(os.path.join(output_root, cls, domain, imgn))

        all_img_num += 1
  
  print('all_img_num: {}, not detected: {}, proportion: {:.3f}'.format(all_img_num, not_detected, all_img_num/not_detected))

def main():
    global args
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    test(args)

if __name__ == '__main__':
    main()

