import csv
import os
import sys
import numpy as np
import pickle

val_root = sys.argv[1]
output = sys.argv[2]

gallery = os.path.join(val_root, "gallery.csv")
probe = os.path.join(val_root, "probe.csv")
img_dir = os.path.join(val_root, 'images')

paths0 = []
with open(probe, "r") as probeFile:
    readerProbe = csv.reader(probeFile)
    for _, item in readerProbe:
        paths0.append(item)

paths1 = []
with open(gallery, 'r') as galleryFile:
    readerGallery = csv.reader(galleryFile)
    for _, item in readerGallery:
        paths1.append(item)

filename = os.path.join(val_root, "ground_truth.csv")
with open(filename) as csvFile:
    pair_idxs = list(csv.reader(csvFile))

bins = []
issame = []
paths = []
for pidx, gidx in pair_idxs:
    pidx = int(pidx)
    gidx = int(gidx)
    ppath = os.path.join(img_dir, paths0[pidx])
    if gidx != -1:
        npath = np.random.choice(paths1)
        npath = os.path.join(img_dir, npath)
        paths += (ppath, npath)
        issame.append(False)
    else:
        gpath = os.path.join(img_dir, paths1[gidx])
        paths += (ppath, gpath)
        issame.append(True)

        prob = np.ones((len(paths1), )) / (len(paths1) - 1)
        prob[gidx] = 0
        npath = np.random.choice(paths1, p = prob)
        npath = os.path.join(img_dir, npath)
        paths += (ppath, npath)
        issame.append(False)

for path in paths:
    with open(path, 'rb') as fin:
        _bin = fin.read()
        bins.append(_bin)

with open(output, 'wb') as f:
    pickle.dump((bins, issame), f, protocol=pickle.HIGHEST_PROTOCOL)
