from __future__ import print_function, division
import argparse
import csv
import numpy as np
import os
import mxnet as mx
from retinaface import RetinaFace
from face_model import FaceModel
import cv2
import mxnet_interface

def parse_args():
    parser = argparse.ArgumentParser(description='Test Face Recogonition task')
    parser.add_argument('--data_rpath', dest='data_rpath', help='relative path of dataset')

    parser.add_argument('--model', dest='model',
                        help='model ckpt path', type=str)
    parser.add_argument('--prediction-file', dest='ppath',
                        help='prediction file path', type=str)
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--save_crop', action="store_true")
    parser.add_argument('--crop_dir', default="../rf-finalA-cropped")
    parser.add_argument('--type', default='cosine', choices=['euclidean','cosine'])
    # RetinaNet: prefix, epoch, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4 = 0.5, vote=False
    parser.add_argument('--pretrained-detector', dest="pdetect",
                        help="detector checkpoint prefix", default="./models/finalR50")
    parser.add_argument('--detector-epoch', dest='depoch', default=0, type=int)
    parser.add_argument('--detector-network', dest="dnet",
                        help="detector config type", default='net3')
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--nocrop', action="store_true")
    parser.add_argument('--align_match', action="store_true")
    args = parser.parse_args()
    return args
    
def detect_or_return_origin(img_path, model, align = False):
    # return RGB image (c, h, w)
    img = cv2.imread(img_path)

    # remenber to delete when in interface.py!
    if img.shape[0] == 224 and img.shape[1] == 224:
        img = cv2.resize(img, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(img, (2,0,1))
        return aligned, True

    new_img = model.get_input(img, threshold=0.02, align=align)

    if new_img is None:
        img = cv2.resize(img, (142, 142))
        b = (142 - 112) // 2
        img = img[b:-b, b:-b, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(img, (2,0,1))
        return aligned, False
    else:
        new_img = cv2.resize(new_img, (112, 112))
        aligned = np.transpose(new_img, (2,0,1))
        return aligned, True

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
    # -------------------------
    # 2. face detection
    path, epoch = args.model.split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, int(epoch))
    model = mx.mod.Module(context = mx.gpu(0), symbol = sym)
    model.bind(data_shapes=[('data', (1, 3, 112, 112))])
    model.set_params(arg_params, aux_params)
    
    # if args.align_match:
    #     detector = RetinaFace('models/R50', 0, 0, args.dnet, args.nms, nocrop=args.nocrop, vote=False)
    #     _, arg_params, aux_params = mx.model.load_checkpoint(args.pdetect, args.depoch)
    #     detector.model.set_params(arg_params, aux_params, allow_missing = True)
    # else:    
    
    detector = RetinaFace(args.pdetect, args.depoch, 0, args.dnet, args.nms, args.nocrop, vote=False)
    fmodel = FaceModel(detector, model)

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
        img0, hit = detect_or_return_origin(img0_path, fmodel, args.align_match)
        prob_imgs.append(img0)
    for _, item in gallery_list:
        img1_path = os.path.join(imgset_rpath, item)
        img1, hit = detect_or_return_origin(img1_path, fmodel, args.align_match)
        gallery_imgs.append(img1)
    # -------------------------
    # 3. face recogonition
    for img0 in prob_imgs:
        probefeature = fmodel.get_feature([img0])
        probeFeature.append(probefeature)

    for img1 in gallery_imgs:
        galleryfeature = fmodel.get_feature([img1])
        galleryFeature.append(galleryfeature)
    # -------------------------
    # 4. prediction
    galleryFeature = mx.ndarray.concat(*galleryFeature, dim=0).asnumpy()
    probeFeature = mx.ndarray.concat(*probeFeature, dim=0).asnumpy()
    # print(galleryFeature.shape, galleryFeature.context)
    filename = os.path.join(data_rpath, "ground_truth.csv")
    csvFile = open(filename, 'r')
    readerC = list(csv.reader(csvFile))

    max_pred = None
    max_auc = 0
    for th in np.arange(0, 0.3, 0.01):
        k = 0
        type1 = 0
        type2 = 0
        metric = mxnet_interface.cal_metric(galleryFeature, probeFeature, args.type, th)
        for item in readerC:
            if metric[int(item[0])] == int(item[1]):
                k += 1
            elif metric[int(item[0])] == -1: # inter distance is small
                type1 += 1
            elif int(item[1]) != -1: # wrong answer: intra distance is larger than inter distance
                type2 += 1
        auc = k / len(metric)
        if auc > max_auc:
            max_pred = metric
            max_auc = auc
        print('threshod: {} , correct: {}, auc: {}, type1 error {}, type2 error {}'.format(th, k, auc, type1, type2))

    if args.vis:
        fake1 = 'vis/fake1'
        fake2 = 'vis/fake2'
        fake3 = 'vis/fake3'
        os.makedirs(fake1, exist_ok=True)
        os.makedirs(fake2, exist_ok=True)
        os.makedirs(fake3, exist_ok=True)
        for item in readerC:
            if max_pred[int(item[0])] != int(item[1]):
                if int(item[1]) == -1:
                    tmp = np.concatenate(
                        (prob_imgs[int(item[0])], gallery_imgs[ max_pred[int(item[0])] ]),
                        axis=-1)
                    tmp = np.transpose(tmp, (1,2,0))
                    cv2.imwrite(
                        os.path.join(fake1, probe_dict[item[0]][-10:]),
                        cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
                elif max_pred[int(item[0])] == -1:
                    tmp = np.concatenate(
                        (prob_imgs[int(item[0])], gallery_imgs[ int(item[1]) ]),
                        axis=-1)
                    tmp = np.transpose(tmp, (1,2,0))
                    cv2.imwrite(
                        os.path.join(fake2, probe_dict[item[0]][-10:]),
                        cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
                elif int(item[1]) != -1:
                    tmp = np.concatenate(
                        (prob_imgs[int(item[0])], gallery_imgs[ max_pred[int(item[0])] ], gallery_imgs[ int(item[1]) ]),
                        axis=-1)
                    tmp = np.transpose(tmp, (1,2,0))
                    cv2.imwrite(
                        os.path.join(fake3, probe_dict[item[0]][-10:]),
                        cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
