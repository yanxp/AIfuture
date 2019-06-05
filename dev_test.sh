export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
TEST_SET="../rf-finalA-cropped/"
#TEST_SET="../af2019-ksyun-finalA-20190518/"

FILE=$1
shift
# python test.py --test-dataset ../af2019-ksyun-finalA-20190518/ --model models/finalmodel.pth $@
python $FILE --test-dataset $TEST_SET $@
