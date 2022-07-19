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


def virtual_main(detector: Detector, test_img=None, test_img_dir=None):
    """
    虚拟读图测试程序

    :param detector: 杂质探测器，需要继承Detector类
    :param test_img: 测试图像，rgb格式的图片或者路径
    :param test_img_dir: 测试图像文件夹
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
        result = detector.predict(img)
        t2 = time.time()
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(img)
        axs[1].imshow(result)
        mask_color = np.zeros_like(img)
        mask_color[result > 0] = (0, 0, 255)
        result_show = cv2.addWeighted(img, 1, mask_color, 0.5, 0)
        axs[2].imshow(result_show)
        plt.title(f'{(t2 - t1) * 1000:.2f} ms')
        plt.show()


if __name__ == '__main__':
    detector = AnonymousColorDetector(file_path='models/ELM_2022-07-18_17-22.mat')
    virtual_main(detector, test_img='data/dataset/img/yangeng.bmp')
