import os
import numpy as np
# img_dir = '/home/yanxiaopeng/codework/dataset/af2019-ksyun-training-20190416/images/'
anno_dir = '/home/yanxiaopeng/codework/dataset/annotations'

boxes = []
fnames = []

for cls in os.listdir(anno_dir):
    if len(cls)<8:
        ppath = os.path.join(anno_dir,cls,'1')
        for anno in os.listdir(ppath):
            annofile = os.path.join(ppath,anno)
            bbox = np.load(annofile)
            imgpath = os.path.join(cls,'1',anno.split('.')[0][:-3]+'.jpg')
            if not os.path.exists('images/' + imgpath):
                imgpath = os.path.join(cls,'1',anno.split('.')[0][:-3]+'.png')
            bbox = bbox[0][:-1]
            """
            img = cv2.imread(imgpath)
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imwrite('vis/'+imgpath.split('/')[-1],img)
            """
            boxes.append(bbox)
            fnames.append(imgpath)
        ppath = os.path.join(anno_dir,cls,'0')
        for anno in os.listdir(ppath):
            annofile = os.path.join(ppath,anno)
            bbox = np.load(annofile)
            imgpath = os.path.join(cls,'0',anno.split('.')[0][:-3]+'.jpg')
            if not os.path.exists('images/'+imgpath):
                imgpath = os.path.join(cls,'0',anno.split('.')[0][:-3]+'.png')
            bbox = bbox[0][:-1]
            """
            img = cv2.imread(imgpath)
            cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imwrite('vis/'+imgpath.split('/')[-1],img)
            h,w,_ = img.shape
            bbox = convert_boxes(bbox,w,h)
            """
            boxes.append(bbox)
            fnames.append(imgpath)

with open('label.txt', 'w') as f:
    for path, box in zip(fnames, boxes):
        s = '# ' + path + '\n'
        f.write(s)
        s = '{:.1f} {:.1f} {:.1f} {:.1f} '.format(box[0], box[1], box[2], box[3])
        s += ' '.join(['-1'] * 16)
        f.write(s)
        f.write('\n')
