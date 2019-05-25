#!/bin/bash

# 根据项目，配置环境变量
source activate afteam_911
export PRETRAINED_MODEL=./models/finalmodel.pth

# 调用interface.py执行测试，其中data_rpath是数据集所在相对路径, team_number是参赛队伍编号，version_number是参赛队伍提交作品的版本号
python interface.py --data_rpath $1 --team_number 911 --version_number 0.1

