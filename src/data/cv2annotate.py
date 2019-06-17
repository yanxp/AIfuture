import os
import numpy as np
import sys
import cv2
img_dir = sys.argv[1] # 'dataset/af2019-ksyun-training-20190416/images/'
label_file = sys.argv[2] # labels.txt
annots = []
fnames = []
face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

for cls in os.listdir(img_dir):
    if len(cls)<8:
        ppath = os.path.join(img_dir,cls,'1')
        for imgp in os.listdir(ppath):
            img = cv2.imread(imgp)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = face_patterns.detectMultiScale(gray, gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT)
            if len(face) == 0:
                continue
            fnames.append(imgp)

            '''x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(imgpath.split('/')[-1],img)
            h,w,_ = img.shape
            return'''
            annots.append(faces[0])
        ppath = os.path.join(anno_dir,cls,'0')
        for imgp in os.listdir(ppath):
            img = cv2.imread(imgp)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = face_patterns.detectMultiScale(gray, gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_FIND_BIGGEST_OBJECT)
            if len(face) == 0:
                continue
            fnames.append(imgp)

            '''x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(imgpath.split('/')[-1],img)
            h,w,_ = img.shape'''
            annots.append(faces[0])

with open(label_file, 'w') as f:
    for path, annot in zip(fnames, annots):
        s = '# ' + path + '\n'
        f.write(s)
        s = '{:.1f} {:.1f} {:.1f} {:.1f} '.format(annot[0], annot[1], annot[2], annot[3])
        s += ' '.join(['-1'] * 16)
        f.write(s)
        f.write('\n')