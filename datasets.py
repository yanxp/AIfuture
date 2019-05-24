# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import random
import shutil
root = '../af2019-ksyun-training-20190416/images/'   # change this root of your own data
def getExample (path , n):
    example_list = []
    return_list = []
    for example in os.listdir(path):
        examplePath = os.path.join(path,example)
        example_list.append(examplePath)
    return_list = list(random.sample(example_list, n))
    return return_list

def getClass (path):
    classPath = path.split('/')[-3]
    return classPath


# print("photo---------------------------------------------------------------------------------------")
pExamleList = []
cExamleList = []
thresholdNum = 6
for person in os.listdir(root):
    if os.path.isfile(os.path.join(root, person)): break
    count = 0
    person_dir = os.path.join(root, person,'1')
    if os.path.isfile(person_dir):break
    for pictures in os.listdir(person_dir):
        count += 1
    if count >= thresholdNum:
        photo = getExample(person_dir , thresholdNum)
    else:
        photo = getExample(person_dir,count)
    pExamleList.append(photo)

    count = 0
    person_dir = os.path.join(root, person,'0')
    for pictures in os.listdir(person_dir):
        count += 1
    if count >= thresholdNum:
        caricature = getExample(person_dir , thresholdNum)
    else:
        caricature = getExample(person_dir,count)
    cExamleList.append(caricature)

triples_train = []
triples_val = []

number = len(pExamleList)
for i in range(number):
    pList = pExamleList[i]
    cList = cExamleList[i]
    # 获得pList的父目录
    pClassName = getClass(pList[0])
    # 判断当前抽取的十个漫画类与当前的照片属于同一个类，则重新抽取
    # 判断当前抽取的十个漫画类与当前的照片属于同一个类，则重新抽取
    flag = True
    while flag:
        cClass = getExample(root, thresholdNum)
        for cClassName in cClass:
            cClassName = cClassName.split('/')[-1]
            if pClassName == cClassName:
                flag = True
                break
            else:
                flag = False
    for p in pList:
        for q in cList:
            for c in cClass:
                c = c+'/0'
                c = getExample(c,1)
                triple = (p,q,c[0])
                if i < 100:
                    triples_train.append(triple)
                else:
                    triples_val.append(triple)

print(len(triples_train))
print(len(triples_val))

save_path3 = '../af2019-ksyun-training-20190416/data_triples_train.pkl'
output3 = open(save_path3, 'wb')
pickle.dump(triples_train, output3)
output3.close()

save_path4 = '../af2019-ksyun-training-20190416/data_triples_val.pkl'
output4 = open(save_path4, 'wb')
pickle.dump(triples_val, output4)
output4.close()
