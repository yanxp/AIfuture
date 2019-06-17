# -*- coding:utf-8 -*-
from __future__ import print_function, division
import time
import argparse
import os

def predict_interface(imgset_rpath: str, gallery_dict: dict, probe_dict: dict) -> [(str, str), ...]:
    """
    imgset_rpath: 数据图片库的相对路径，例如数据集的相对路径为test_set/testB，那么图片库的相对路径为test_set/testB/images
    在run_test.sh中已经设置了数据集的默认地址

    gallery_dict:  {gallery_pic_id: pic_rpath,...}，类型为dict。
    其中gallery_pic_id为图片在gallery库中的id，类型为str，下同；
    pic_rpath 为图片在图片库中的相对路径，类型为str，例如test_set/testB/images/dd/dd7e12af086264e1b13d8a788857077d.jpg

    probe_dict:    {probe_pic_id: pic_rpath,...}，类型为dict。
    其中probe_pic_id为图片在probe库中的id，类型为str，下同；
    pic_rpath 为图片在图片库中的相对路径，类型为str，例如test_set/testB/images/65/65125c57356c12c916b00afcd572b1ec.jpg

    :return [(probe_pic_id, gallery_pic_id),...]，类型为list。
    其中元素是由probe_pic_id和gallery_pic_id组成的tuple。
    """
    framework = os.getenv('FRAMEWORK')
    if framework == 'mxnet':
        import mxnet_interface
        return mxnet_interface.predict_interface(imgset_rpath, gallery_dict, probe_dict)
    elif framework == 'multiprocess':
        import mult_interface
        return mult_interface.predict_interface(imgset_rpath, gallery_dict, probe_dict)
    else:
        import pytorch_interface
        return pytorch_interface.predict_interface(imgset_rpath, gallery_dict, probe_dict)
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='Welcome to ACG Face contest! Please input your argument according to usage.'
    )
    parser.add_argument(
        '--data_rpath',
        dest='data_rpath',
        help='relative path of dataset',
        type=str
    )
    parser.add_argument(
        '--team_number',
        dest='team_number',
        help='the number of team',
        type=str
    )
    parser.add_argument(
        '--version_number',
        dest='version_number',
        help='version number of work submitted',
        type=str
    )
    args = parser.parse_args()
    return args


def run_test():
    args = parse_args()
    data_rpath = args.data_rpath
    team_number = args.team_number
    version_number = args.version_number

    gallery_dict = {}
    probe_dict = {}

    probe_csv = os.path.join(data_rpath, 'probe.csv')
    for line in open(probe_csv):
        probe_pic_id = line.strip().split(',')[0]
        probe_pic_url = line.strip().split(',')[1]
        probe_dict[probe_pic_id] = probe_pic_url

    gallery_csv = os.path.join(data_rpath, 'gallery.csv')
    for line in open(gallery_csv):
        gallery_pic_id = line.strip().split(',')[0]
        gallery_pic_url = line.strip().split(',')[1]
        gallery_dict[gallery_pic_id] = gallery_pic_url

    t_start = time.time()

    result_list = predict_interface(
        imgset_rpath=os.path.join(data_rpath, 'images'),
        gallery_dict=gallery_dict,
        probe_dict=probe_dict
    )

    t_end = time.time()
    duration = t_end - t_start

    result_str = 'Team_number: {}, version_number: {}, total_time: {:.1f}seconds.\n'\
        .format(team_number, version_number, duration)

    for result_i in result_list:
        result_str += '{}, {}\n'.format(result_i[0], result_i[1])

    with open('result_%s_%s.csv' % (team_number, version_number), 'w') as output_file:
        output_file.write(result_str)


if __name__ == '__main__':
    run_test()
