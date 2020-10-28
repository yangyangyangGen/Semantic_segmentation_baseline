from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from functools import wraps, partial


def global_config():
    return {
        "data_root": r"D:\workspace\DataSets\seg\cityscapes\cityscapes",
        "train_list": r"D:\workspace\DataSets\seg\cityscapes\pix2pix_train_list",
        "test_list": r"D:\workspace\DataSets\seg\cityscapes\pix2pix_test_list",
    }


global_dict = global_config()



if __name__ == '__main__':
    pass






