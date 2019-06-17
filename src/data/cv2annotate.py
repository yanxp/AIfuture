import os
import numpy as np
import sys
import cv2
img_dir = sys.argv[1] # 'dataset/af2019-ksyun-training-20190416/images/'
label_file = sys.argv[2] # labels.txt
annots = []
fnames = []
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000.caffemodel")

def detect_img(net, img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0,
        (300, 300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    h, w, c = img.shape
    detections = detections[0,0]
    if len(detections) == 0:
        return None
    confidence = detections[0][2]
    if confidence < 0.6:
        return None
    box = detections[0, 3:7] * np.array([w,h,w,h])
    return np.array([box[0], box[1], box[2]-box[0], box[3]-box[1]])

with open(label_file, 'w') as f:
    for cls in os.listdir(img_dir):
        if len(cls)<8:
            ppath = os.path.join(img_dir,cls,'1')
            for imgp in os.listdir(ppath):
                imgpp = os.path.join(ppath, imgp)
                img = cv2.imread(imgpp)
                face = detect_img(net, img)
                if face is None:
                    continue
                '''x, y, w, h = face
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.imwrite(imgp.split('/')[-1],img)
                h,w,_ = img.shape
                sys.exit(0)'''
                s = '# ' + os.path.join(cls, '1', imgp) + '\n'
                f.write(s)
                annot = face
                s = '{:.1f} {:.1f} {:.1f} {:.1f} '.format(annot[0], annot[1], annot[2], annot[3])
                s += ' '.join(['-1'] * 16)
                f.write(s)
                f.write('\n')
            ppath = os.path.join(img_dir,cls,'0')
            for imgp in os.listdir(ppath):
                imgpp = os.path.join(ppath, imgp)
                img = cv2.imread(imgpp)
                face = detect_img(net, img)
                if face is None:
                    continue
                x, y, w, h = face
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.imwrite(imgp.split('/')[-1],img)
                h,w,_ = img.shape
                sys.exit(0)
                s = '# ' + os.path.join(cls,'0',imgp) + '\n'
                f.write(s)
                annot = face
                s = '{:.1f} {:.1f} {:.1f} {:.1f} '.format(annot[0], annot[1], annot[2], annot[3])
                s += ' '.join(['-1'] * 16)
                f.write(s)
                f.write('\n')
