# 环境配置要求

* python3.6

# 参考的第三方程序或论文
* (Triplet network)[https://arxiv.org/abs/1412.6622]
* (insightface)[https://github.com/deepinsight/insightface]

# 第三方开源模型
* VGG_FACE.t7(最后未使用): (下载链接)[http://www.robots.ox.ac.uk/~vgg/software/vgg_face/]
* SSD：(下载链接)[https://anonfile.com/W7rdG4d0b1/face_detector.rar]，该模型是由opencv提供的预训练模型。
* ImageNet resnet-50：(下载链接)[https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA]，该模型是由insightface提供的预训练模型。
* RetinaFace-R50: (下载链接)[https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ]，该模型是由insightface提供的预训练模型。
* LResNet100E-IR,ArcFace@ms1m-refine-v2：(下载链接)[https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA]，该模型是由insightface提供的预训练模型。
# 运行过程
## 获得测试结果
**安装环境**
```
./install.sh
```
**运行测试**
```
./run_test.sh
```

## 训练全过程
目录变量设置:
* CODE_ROOT：代码所在根目录
* TRAIN_ROOT：训练集所在根目录
* TEST_ROOT：初赛测试集testA所在根目录，其中的list.csv为本队**手工标注**的配对信息。

### 训练检测器
#### 准备数据
首先，需要用opencv提供的预训练模型标注部分训练图片，假设我们现在位于`$CODE_ROOT`目录下，运行如下命令：
```Bash
$> cd src/data
$> python cv2annotate.py $TRAIN_ROOT/images label.txt
$> cd $CODE_ROOT/retinaface/data
# 若所需训练文件 images 和 label.txt已经存在于aifuture/train下，则跳过下面几步
$> mkdir -p aifuture/train 
$> cd aifuture/train
$> ln -s $TRAIN_ROOT/images images
$> mv $CODE_ROOT/src/data/label.txt .
```
#### 训练RetinaFace
现在，假设我们正在`$CODE_ROOT/retinaface`文件夹中。并且已经下载好ImageNet resnet-50和RetinaFace-R50，放置在`$CODE_ROOT/models`文件夹中。
```Bash
$> CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --checkpoint ../models/R50 --dataset_path data/aifuture/ --pretrained ../models/resnet-50
```
本次比赛提交的版本，是训练了4个epoch过后得到的，我们将其存放于`$CODE_ROOT/models/testR50-0004.params`和`$CODE_ROOT/models/testR50-symbol.json`.
#### 合并landmark分支
在`$CODE_ROOT`目录下，运行`python combineRetina.py`，获得`$CODE_ROOT/models/finalR50`。具体的路径参数请参见脚本。

### 训练人脸识别模型
#### 准备数据
首先，我们使用训练得到的RetinaFace对训练集进行人脸对齐。
```Bash
$> cd $CODE_ROOT
$> python -m retinaface.crop_aifuture --data_rpath $TRAIN_ROOT/images/ --prefix ./models/finalR50 --output $CROP_TRAIN_ROOT
# CROP_TRAIN_ROOT是剪裁后图片的根目录，目录结构与$TRAIN_ROOT相同
# 剪裁验证集testA，验证集是直接覆盖原图片的，因此最好备份一下testA
$> python -m retinaface.crop_aifuture --data_rpath $TEST_ROOT/images/ --prefix models/finalR50 --dataset testA
```

接着，生成训练所需的mxnet record格式文件：
```Bash
# 生成.lst
$> cd $CODE_ROOT/src/data
$> python dir2lst_ytf.py $CROP_TRAIN_ROOT/images/ > $CROP_TRAIN_ROOT/train.lst
# 获得.rec
$> python face2rec2.py  --num-thread 8 $CROP_TRAIN_ROOT
# 获得验证集
$> python testA2pair.py $TEST_ROOT $CROP_TRAIN_ROOT/pair.bin
```
#### 训练模型
```Bash
$> cd $CODE_ROOT/recogonition
$> CUDA_VISIBLE_DEVICES=1,2,3,4 python -u train.py --dataset aifuture --verbose 200 --lr 0.001 --ckpt 2
```
最终，提交保存的第三个模型`$CODE_ROOT/recogonition/models/r100-arcface-aifuture/model-0003.params`

# 比赛版本日志
* 5月25日 版本 0.1
直接使用初赛训练出的模型做流程测试，使用特征归一化后的欧式距离作为判断是否是同一个人的依据。对于所有gallery，将本probe的特征和gallery的特征计算欧式距离后排序，计算最小值和中位数之差，如果差大于0.375则判定匹配有效，输出距离最小的gallery id，否则输出-1.

* 5月27日 版本 0.2
在原来的模型上增加了人脸检测器，该检测器尚未对漫画域进行finetune，仅作为环境测试使用。增加该检测器后，之前的阈值由0.375变为0.4

* 6月1日 版本 0.2.1
使用opencv进行的人脸标注训练+RetinaFace的[公开模型](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ)，对于漫画域进行finetune。在当前代码设置下训练得到的testR50-0004，在finalA中能够实现照片全检测，漫画有4张检测不出来。

* 6月5日 版本0.3
使用softmax+triplet loss训练新的模型，调整了判定阈值

* 6月7日 版本1.0
使用resetnet[公开模型](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA)+arcface进行训练，调整了判定标准为cos，调整判定阈值。使用triplet微调时已经无法选出triple，说明仅仅分类loss就可以了
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py --dataset aifuture --loss arcface --lr 0.001
# per-batch-size 32 verbose 500 获得r100arc-0004
```
* 6月8日 版本1.1
提交试用atriplet loss微调的模型，测试成绩变差，估计过拟合。

* 6月9日 版本1.2
发现昨天的训练对于dropout设置错误，重新训练了两个模型，根据可视化结果选择r100arc-0006.params
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py --dataset aifuture --loss arcface --lr 0.001 --pretrained ../models/..s/model-r100-ii/model --ckpt 2 --verbose 500
```
实验停止时，训练集精度大约为95%。另外0002的表现也不错，但是可视化结果看起来不太靠谱，并且训练集精度只有45%左右，估计只是运用finetune之前学到的特征来进行匹配。

* 6月10日 版本1.3
修改了一些操作到mxnet上进行加速，添加flip_match，将测试margin改为30( 在所提交的 0004 模型上表现更好? )

* 6月12日 版本1.5
使用R50的face landmark以及testR50 的检测参数完成人脸对齐，使用人脸关键点对齐后的数据进行训练和检测


