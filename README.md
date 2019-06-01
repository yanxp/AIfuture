# 环境配置要求

* python3.6

# 参考的第三方程序或论文
* (Triplet network)[https://arxiv.org/abs/1412.6622]
* insightface
# 第三方开源模型
* VGG_FACE.t7: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
* mtcnn
# 设计思路
* 5月25日 版本 0.1
直接使用初赛训练出的模型做流程测试，使用特征归一化后的欧式距离作为判断是否是同一个人的依据。对于所有gallery，将本probe的特征和gallery的特征计算欧式距离后排序，计算最小值和中位数之差，如果差大于0.375则判定匹配有效，输出距离最小的gallery id，否则输出-1.

* 5月27日 版本 0.2
在原来的模型上增加了人脸检测器，该检测器尚未对漫画域进行fineturn，仅作为环境测试使用。增加该检测器后，之前的阈值由0.375变为0.4

* 6月1日 版本 0.2.1
使用opencv进行的人脸标注训练+RetinaFace的[公开模型](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ)，对于漫画域进行finetune。在当前代码设置下训练得到的testR50-0004，在finalA中能够实现照片全检测，漫画有4张检测不出来。


# 运行过程
## 获得测试结果
**安装环境**
```
./install.sh
```
**运行测试**
```
./run_test.sh {到测试集的相对路径}
# 例子
./run_test.sh ../af2019-ksyun-finalA-20190518
```

## 训练
Train:

	1. download the VGG_FACE.t7 http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
	
    2. run dataset.py to generate the train and val .pkl files
    
    3. python train_triplet.py to train the models and default save models in triplet_models dir

