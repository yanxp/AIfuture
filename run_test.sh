#!/bin/bash

# 根据项目，配置环境变量

# >>> conda init >>>

__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"

if [ $? -eq 0 ]; then

    \eval "$__conda_setup"

else

    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then

        . "$HOME/anaconda3/etc/profile.d/conda.sh"

        CONDA_CHANGEPS1=false conda activate base

    else

        \export PATH="$PATH:$HOME/anaconda3/bin"

    fi

fi

unset __conda_setup

# <<< conda init <<<
conda activate afteam_911
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PRETRAINED_MODEL='models/r100align/model,3'
export ALIGN_MATCH=1
export BATCH_SIZE=8
export FRAMEWORK='mxnet'
#export FRAMEWORK='multiprocess'
#export FRAMEWORK='pytorch'

# 调用interface.py执行测试，其中data_rpath是数据集所在相对路径, team_number是参赛队伍编号，version_number是参赛队伍提交作品的版本号
#python interface.py --data_rpath $1 --team_number 911 --version_number 0.2
python interface.py --data_rpath ../B_test_set/testB --team_number 911 --version_number 1.5
