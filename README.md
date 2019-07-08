# The Code is for  <a href="https://ai.futurelab.tv/contest_detail/4"> AI Future Comic Face Recognition Contest </a> of the Second Place.

### Acknowledgement

Thanks for the contribution of <a href="https://github.com/MintYiqingchen">Xiaoxi Wang</a>.

### License

For Academic Research Use Only!

### Requirements

python 3.6

opencv 

mxnet-cu92

### Usage

### 1. 训练检测器

准备数据

首先，需要用 opencv 提供的预训练模型标注部分训练图片，假设我们现在位于
$CODE_ROOT 目录下，运行如下命令：

$> cd src/data$> python cv2annotate.py $TRAIN_ROOT/images label.txt

$> cd $CODE_ROOT/retinaface/data

$> mkdir -p aifuture/train

$> cd aifuture/train

$> ln -s $TRAIN_ROOT/images images

$> mv $CODE_ROOT/src/data/label.txt .

这一步完成后， $CODE_ROOT/retinaface/data/aifuture 目录的结构为

----$CODE_ROOT/retinaface/data/aifuture

|__ train/

|__images/

|__label.txt

训练 RetinaFace

假设进入了$CODE_ROOT/retinaface 文件夹中。并且已经下载好 ImageNet
resnet-50 和 RetinaFace-R50，放置在$CODE_ROOT/models 文件夹中。

$> CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --checkpoint ../models/R50 --
dataset_path data/aifuture/ --pretrained ../models/resnet-50

合并 landmark 分支

在$CODE_ROOT 目录下，运行 python combineRetina.py，获得
$CODE_ROOT/models/finalR50。具体的路径参数请参见脚本。

### 2. 训练人脸识别模型

首先使用训练得到的 RetinaFace 对训练集进行人脸对齐。

$> cd $CODE_ROOT

$> python -m retinaface.crop_aifuture --data_rpath $TRAIN_ROOT/images/ --
prefix ./models/finalR50 --output $CROP_TRAIN_ROOT

$> python -m retinaface.crop_aifuture --data_rpath $TEST_ROOT/images/ --prefix
models/finalR50 --dataset testA

接着，生成训练所需的 mxnet record 格式文件：

$> cd $CODE_ROOT/src/data

$> python dir2lst_ytf.py $CROP_TRAIN_ROOT/images/ > $CROP_TRAIN_ROOT/train.lst # 生成.lst

$> python face2rec2.py --num-thread 8 $CROP_TRAIN_ROOT # 获得.rec

$> python testA2pair.py $TEST_ROOT $CROP_TRAIN_ROOT/pair.bin # 获得验证集

训练模型

$> cd $CODE_ROOT/recogonition

$> CUDA_VISIBLE_DEVICES=1,2,3,4 python -u train.py --dataset aifuture --verbose 200
--lr 0.001 --ckpt 2

最终保存第三个模型$CODE_ROOT/recogonition/models/r100-arcfaceaifuture/model-0003.params