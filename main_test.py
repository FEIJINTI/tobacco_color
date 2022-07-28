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

import transmit
from config import Config
from models import Detector, AnonymousColorDetector, ManualTree, SpecDetector, RgbDetector
from utils import read_labeled_img, size_threshold, natural_sort


class TestMain:
    def __init__(self):
        self._spec_detector = SpecDetector(blk_model_path=Config.blk_model_path,
                                           pixel_model_path=Config.pixel_model_path)
        self._rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                                         background_model_path=Config.rgb_background_model_path)

    def pony_run(self, test_path, test_spectra=False, test_rgb=False,
                 convert=False):
        """
        虚拟读图测试程序

        :param test_path: 测试文件夹或者图片
        :param test_spectra: 是否测试光谱
        :param test_rgb: 是否测试rgb
        :param convert: 是否进行格式转化
        :return:
        """
        if os.path.isdir(test_path):
            rgb_file_names, spec_file_names = [[file_name for file_name in os.listdir(test_path) if
                                                file_name.startswith(file_type)] for file_type in ['rgb', 'spec']]
            rgb_file_names, spec_file_names = natural_sort(rgb_file_names), natural_sort(spec_file_names)
        else:
            if test_spectra:
                with open(test_path, 'rb') as f:
                    data = f.read()
                spec_img = transmit.BeforeAfterMethods.spec_data_post_process(data)
                _ = self.test_spec(spec_img=spec_img)
            elif test_rgb:
                with open(test_path, 'rb') as f:
                    data = f.read()
                rgb_img = transmit.BeforeAfterMethods.rgb_data_post_process(data)
                _ = self.test_rgb(rgb_img)
            return
        for rgb_file_name, spec_file_name in zip(rgb_file_names, spec_file_names):
            if test_spectra:
                with open(os.path.join(test_path, spec_file_name), 'rb') as f:
                    data = f.read()
                spec_img = transmit.BeforeAfterMethods.spec_data_post_process(data)
                spec_mask = self.test_spec(spec_img, img_name=spec_file_name)
            if test_rgb:
                with open(os.path.join(test_path, rgb_file_name), 'rb') as f:
                    data = f.read()
                rgb_img = transmit.BeforeAfterMethods.rgb_data_post_process(data)
                rgb_mask = self.test_rgb(rgb_img, img_name=rgb_file_name)
            if test_rgb and test_spectra:
                self.merge(rgb_img=rgb_img, rgb_mask=rgb_mask,
                           spec_img=spec_img[..., [21, 3, 0]], spec_mask=spec_mask,
                           file_name=rgb_file_name)

    def test_rgb(self, rgb_img, img_name):
        rgb_mask = self._rgb_detector.predict(rgb_img)
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(rgb_img)
        axs[0].set_title(f"rgb img {img_name}")
        axs[1].imshow(rgb_mask)
        axs[1].set_title('rgb mask')
        plt.show()
        return rgb_mask

    def test_spec(self, spec_img, img_name):
        spec_mask = self._spec_detector.predict(spec_img)
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(spec_img[..., [21, 3, 0]])
        axs[0].set_title(f"spec img {img_name}")
        axs[1].imshow(spec_mask)
        axs[1].set_title('spec mask')
        plt.show()
        return spec_mask

    @staticmethod
    def merge(rgb_img, rgb_mask, spec_img, spec_mask, file_name):
        mask_result = (spec_mask | rgb_mask).astype(np.uint8)
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title(file_name)
        axs[0, 0].imshow(rgb_img)
        axs[1, 0].imshow(spec_img)
        axs[2, 0].imshow(mask_result)
        axs[0, 1].imshow(rgb_mask)
        axs[1, 1].imshow(spec_mask)
        axs[2, 1].imshow(mask_result)
        plt.show()
        return mask_result


if __name__ == '__main__':
    testor = TestMain()
    testor.pony_run(test_path=r'E:\zhouchao\728-tobacco\728-1-3',
                    test_rgb=True, test_spectra=True)
