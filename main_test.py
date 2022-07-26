# -*- codeing = utf-8 -*-
# Time : 2022/7/19 10:49
# @Auther : zhouchao
# @File: main_test.py
# @Software:PyCharm
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from models import Detector, AnonymousColorDetector, ManualTree
from utils import read_labeled_img, size_threshold


def pony_run(test_img=None, test_img_dir=None, test_spectra=False, test_rgb=False):
    """
    虚拟读图测试程序

    :param test_img: 测试图像，rgb格式的图片或者路径
    :param test_img_dir: 测试图像文件夹
    :param test_spectra: 是否测试光谱
    :param test_rgb: 是否测试rgb
    :return:
    """
    if (test_img is not None) or (test_img_dir is not None):
        threshold = Config.spec_size_threshold
        rgb_threshold = Config.rgb_size_threshold
        manual_tree = ManualTree(blk_model_path=Config.blk_model_path, pixel_model_path=Config.pixel_model_path)
        tobacco_detector = AnonymousColorDetector(file_path=Config.rgb_tobacco_model_path)
        background_detector = AnonymousColorDetector(file_path=Config.rgb_background_model_path)
    if test_img is not None:
        if isinstance(test_img, str):
            img = cv2.imread(test_img)[:, :, ::-1]
        elif isinstance(test_img, np.ndarray):
            img = test_img
        else:
            raise TypeError("test img should be np.ndarray or str")
    if test_img_dir is not None:
        image_names = [img_name for img_name in os.listdir(test_img_dir) if img_name.endswith('.png')]
        for image_name in image_names:
            rgb_data = cv2.imread(os.path.join(test_img_dir, image_name))[..., ::-1]
            # 识别
            t1 = time.time()
            if test_spectra:
                # spectra part
                pixel_predict_result = manual_tree.pixel_predict_ml_dilation(data=img_data, iteration=1)
                blk_predict_result = manual_tree.blk_predict(data=img_data)
                mask = (pixel_predict_result & blk_predict_result).astype(np.uint8)
                mask_spec = size_threshold(mask, Config.blk_size, threshold)
            if test_rgb:
                # rgb part
                rgb_data = tobacco_detector.pretreatment(rgb_data)
                background = background_detector.predict(rgb_data)
                tobacco = tobacco_detector.predict(rgb_data)
                tobacco_d = tobacco_detector.swell(tobacco)
                rgb_predict_result = 1 - (background | tobacco_d)
                mask_rgb = size_threshold(rgb_predict_result, Config.blk_size, Config.rgb_size_threshold)
                fig, axs = plt.subplots(5, 1, figsize=(12, 10), constrained_layout=True)
                axs[0].imshow(rgb_data)
                axs[0].set_title("rgb raw data")
                axs[1].imshow(background)
                axs[1].set_title("background")
                axs[2].imshow(tobacco)
                axs[2].set_title("tobacco")
                axs[3].imshow(rgb_predict_result)
                axs[3].set_title("1 - (background + dilate(tobacco))")
                axs[4].imshow(mask_rgb)
                axs[4].set_title("final mask")
                plt.show()

            mask_result = (mask | mask_rgb).astype(np.uint8)
            # mask_result = rgb_predict_result
            mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
            t2 = time.time()
            print(f'rgb len = {len(rgb_data)}')


if __name__ == '__main__':
    pony_run(test_img_dir=r'E:\zhouchao\725data', test_rgb=True)
