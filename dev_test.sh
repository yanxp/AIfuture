export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python test.py --test-dataset /home/yanxiaopeng/codework/dataset/af2019-ksyun-finalA-20190518/ --model models/finalmodel.pth $@
