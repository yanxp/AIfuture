from __future__ import print_function

import argparse
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from .rcnn.logger import logger
from .retinaface import RetinaFace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from face_model import FaceModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test widerface by retinaface detector')
    # general
    parser.add_argument('--network', help='network name', default='net3', type=str)
    parser.add_argument('--data_rpath', help='dataset name', default='', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default='', type=str)
    parser.add_argument('--epoch', help='model to test with', default=4, type=int)
    parser.add_argument('--output', help='output folder', default='/home/chenriquan/aifuture/rf-cropped-training', type=str)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--nocrop', action="store_true")
    parser.add_argument('--image_size', type=int, default=112, help="output image size")
    parser.add_argument('--align', action="store_true")
    parser.add_argument('--dataset', default='train', choices=['train', 'finalA', 'testA'])
    parser.add_argument('--lst', default='', help='use this to generate lst file')
    args = parser.parse_args()
    return args

detector = None
args = None

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
def crop_train(args, fmodel):
  all_img_num = 0
  not_detected = 0
  
  output_root = os.path.join(args.output, 'images')
  if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
  remain_path = []
  i = 0
  if len(args.lst) > 0:
    lst_file = open(args.lst, 'w')
  for cls in os.listdir(args.data_rpath):
    for domain in ['0', '1']:
      path = os.path.join(args.data_rpath, cls, domain)
      for imgn in os.listdir(path):
        imgp = os.path.join(path, imgn)
        img = cv2.imread(imgp)
        if len(args.lst) == 0:
          new_img = fmodel.get_input(img, threshold=0.02, align=args.align)

          tmp = os.path.join(output_root, cls, domain)
          os.makedirs(tmp, exist_ok=True)
          tmp = os.path.join(tmp, imgn)
          if new_img is None:
            not_detected += 1
            remain_path.append(tmp)
            img = cv2.resize(img, (args.image_size + 30, args.image_size + 30))
            b = 15
            img = img[b:-b, b:-b, :]
            cv2.imwrite(tmp, img)
          else:
            new_img = cv2.resize(new_img, (args.image_size, args.image_size))
            cv2.imwrite(tmp, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

          print('save in ', tmp)
        else:
          annots = fmodel.get_annots(img, thershold=0.02)
          s = "0\t" + imgp + '\t' + str(i)
          print(s)
          if annots is not None:
            s += '\t' + '\t'.join(map(int, annots[0]))
            landmark = annots[1].reshape((10,))
            s += '\t' + '\t'.join(map(int, landmark)) + '\n'
            lst_file.write(s)
          else:
            lst_file.write(s + '\n')
            not_detected += 1
            remain_path.append(tmp)
        all_img_num += 1
    
    i += 1
  print('all_img_num: {}, not detected: {}, proportion: {:.3f}'.format(all_img_num, not_detected, not_detected/all_img_num))

  with open(os.path.join(args.output, 'not_detected.txt'), 'w') as f:
    f.write('\n'.join(remain_path))

def crop_testA(args, fmodel):
  gallery = os.path.join(args.data_rpath, "list.csv")
  img_dir = os.path.join(args.data_rpath, 'images')
  with open(gallery, "r") as probeFile:
    lines = list(probeFile.readlines())[1:]
  for line in lines:
    _, path1, path2, label = line.split(',')
    path1 = os.path.join(img_dir, path1)
    path2 = os.path.join(img_dir, path2)
    for path in (path1, path2):
      img = cv2.imread(path)
      new_img = fmodel.get_input(img, threshold=0.02, align=args.align)

      if new_img is None:
        img = cv2.resize(img, (args.image_size + 30, args.image_size + 30))
        b = 15
        img = img[b:-b, b:-b, :]
        cv2.imwrite(path, img)
      else:
        new_img = cv2.resize(new_img, (args.image_size, args.image_size))
        cv2.imwrite(path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
      print('save in ', path)
      
def test(args):
  print('test with', args)
  global detector
  
  detector = RetinaFace(args.prefix, args.epoch, 0, args.network, args.nms, nocrop=args.nocrop, vote=False)
  fmodel = FaceModel(detector)
  if args.dataset == 'train':
    crop_train(args, fmodel)
  elif args.dataset == 'finalA':
    pass
  elif args.dataset == 'testA':
    crop_testA(args, fmodel)
    
def main():
    global args
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    test(args)

if __name__ == '__main__':
    main()

