import csv
import os
import sys
import numpy as np
import pickle

val_root = sys.argv[1]
output = sys.argv[2]

gallery = os.path.join(val_root, "list.csv")
img_dir = os.path.join(val_root, 'images')

with open(gallery, "r") as probeFile:
    lines = list(probeFile.readlines())[1:]

bins = []
issame = []
paths = []
for line in lines:
    _, path1, path2, label = line.split(',')
    path1 = os.path.join(img_dir, path1)
    path2 = os.path.join(img_dir, path2)
    paths += (path1, path2)
    if int(label) == 0:
        issame.append(False)
    else:
        issame.append(True)
    print(issame[-1], label)

for path in paths:
    with open(path, 'rb') as fin:
        _bin = fin.read()
        bins.append(_bin)

with open(output, 'wb') as f:
    pickle.dump((bins, issame), f, protocol=pickle.HIGHEST_PROTOCOL)
