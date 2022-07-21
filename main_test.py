# -*- codeing = utf-8 -*-
# Time : 2022/7/19 10:49
# @Auther : zhouchao
# @File: main_test.py
# @Software:PyCharm
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from models import Detector, AnonymousColorDetector
from utils import read_labeled_img


def virtual_main(detector: Detector, test_img=None, test_img_dir=None, test_model=False):
    """
    虚拟读图测试程序

    :param detector: 杂质探测器，需要继承Detector类
    :param test_img: 测试图像，rgb格式的图片或者路径
    :param test_img_dir: 测试图像文件夹
    :param test_model: 是否进行模型约束性测试
    :return:
    """
    if test_img is not None:
        if isinstance(test_img, str):
            img = cv2.imread(test_img)[:, :, ::-1]
        elif isinstance(test_img, np.ndarray):
            img = test_img
        else:
            raise TypeError("test img should be np.ndarray or str")
        t1 = time.time()
        img = cv2.resize(img, (1024, 256))
        t2 = time.time()
        result = 1 - detector.predict(img)
        t3 = time.time()
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(img)
        axs[1].imshow(result)
        mask_color = np.zeros_like(img)
        mask_color[result > 0] = (0, 0, 255)
        result_show = cv2.addWeighted(img, 1, mask_color, 0.5, 0)
        axs[2].imshow(result_show)
        axs[0].set_title(
            f' resize {(t2 - t1) * 1000:.2f} ms, predict {(t3 - t2) * 1000:.2f} ms, total {(t3 - t1) * 1000:.2f} ms')
        plt.show()
    if test_model:
        data_dir = "data/dataset"
        color_dict = {(0, 0, 255): "yangeng"}
        dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)
        ground_truth = dataset['yangeng']
        world_boundary = np.array([0, 0, 0, 255, 255, 255])
        detector.visualize(world_boundary, sample_size=50000, class_max_num=5000, ground_truth=ground_truth)


if __name__ == '__main__':
    detector = AnonymousColorDetector(file_path='dt_2022-07-20_14-40.model')
    virtual_main(detector,
                 test_img=r'C:\Users\FEIJINTI\Desktop\720\binning1\tobacco\Image_2022_0720_1354_46_472-003051.bmp',
                 test_model=True)
    virtual_main(detector,
                 test_img=r'C:\Users\FEIJINTI\Desktop\720\binning1\tobacco\Image_2022_0720_1354_46_472-003051.bmp',
                 test_model=True)
    virtual_main(detector,
                 test_img=r'C:\Users\FEIJINTI\Desktop\720\binning1\tobacco\Image_2022_0720_1354_46_472-003051.bmp',
                 test_model=True)
