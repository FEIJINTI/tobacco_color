# -*- codeing = utf-8 -*-
# Time : 2022/7/19 10:49
# @Auther : zhouchao
# @File: main_test.py
# @Software:PyCharm
import itertools
import logging
import os
import time
import socket
import typing

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
                 convert=False, get_delta=False):
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
                _ = self.test_spec(spec_img=spec_img, img_name=test_path)
            elif test_rgb:
                with open(test_path, 'rb') as f:
                    data = f.read()
                rgb_img = transmit.BeforeAfterMethods.rgb_data_post_process(data)
                _ = self.test_rgb(rgb_img, img_name=test_path)
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
                if get_delta:
                    spec_cv = np.clip(spec_img[..., [21, 3, 0]], a_min=0, a_max=1) * 255
                    spec_cv = spec_cv.astype(np.uint8)
                    delta = self.calculate_delta(rgb_img, spec_cv)
                    print(delta)
                self.merge(rgb_img=rgb_img, rgb_mask=rgb_mask,
                           spec_img=spec_img[..., [21, 3, 0]], spec_mask=spec_mask,
                           rgb_file_name=rgb_file_name, spec_file_name=spec_file_name)

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
    def merge(rgb_img, rgb_mask, spec_img, spec_mask, rgb_file_name, spec_file_name):
        mask_result = (spec_mask | rgb_mask).astype(np.uint8)
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title(rgb_file_name)
        axs[0, 0].imshow(rgb_img)
        axs[1, 0].imshow(spec_img)
        axs[1, 0].set_title(spec_file_name)
        axs[2, 0].imshow(mask_result)
        axs[0, 1].imshow(rgb_mask)
        axs[1, 1].imshow(spec_mask)
        axs[2, 1].imshow(mask_result)
        plt.show()
        return mask_result

    def calculate_delta(self, rgb_img, spec_img, search_area_size=(400, 200), eps=1):
        rgb_grey, spec_grey = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(spec_img, cv2.COLOR_RGB2GRAY)
        _, rgb_bin = cv2.threshold(rgb_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, spec_bin = cv2.threshold(spec_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        spec_bin = cv2.resize(spec_bin, dsize=(rgb_bin.shape[1], rgb_bin.shape[0]))
        search_area = np.zeros(search_area_size)
        for x in range(0, search_area_size[0], eps):
            for y in range(0, search_area_size[1], eps):
                delta_x, delta_y = x - search_area_size[0] // 2, y - search_area_size[1] // 2
                rgb_cross_area = self.get_cross_area(rgb_bin, delta_x, delta_y)
                spce_cross_area = self.get_cross_area(spec_bin, -delta_x, -delta_y)
                response_altitude = np.sum(np.sum(rgb_cross_area & spce_cross_area))
                search_area[x, y] = response_altitude
        delta = np.unravel_index(np.argmax(search_area), search_area.shape)
        delta = (delta[0] - search_area_size[1] // 2, delta[1] - search_area_size[1] // 2)
        delta_x, delta_y = delta

        rgb_cross_area = self.get_cross_area(rgb_bin, delta_x, delta_y)
        spce_cross_area = self.get_cross_area(spec_bin, -delta_x, -delta_y)

        human_word = "SPEC is " + str(abs(delta_x)) + " pixels "
        human_word += 'after' if delta_x >= 0 else ' before '
        human_word += "RGB and " + str(abs(delta_y)) + " pixels "
        human_word += "right " if delta_y >= 0 else "left "
        human_word += "the RGB"

        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(rgb_img)
        axs[0].set_title("RGB img")
        axs[1].imshow(spec_img)
        axs[1].set_title("spec img")
        axs[2].imshow(rgb_cross_area & spce_cross_area)
        axs[2].set_title("cross part")
        plt.suptitle(human_word)
        plt.show()

        print(human_word)
        return delta

    @staticmethod
    def get_cross_area(img_bin, delta_x, delta_y):
        if delta_x >= 0:
            cross_area = img_bin[delta_x:, :]
        else:
            cross_area = img_bin[:delta_x, :]
        if delta_y >= 0:
            cross_area = cross_area[:, delta_y:]
        else:
            cross_area = cross_area[:, :delta_y]
        return cross_area


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run image test or ')
    tester = TestMain()
    tester.pony_run(test_path=r'/home/lzy/2022.7.30/tobacco_v1_0/saved_img/',
                    test_rgb=False, test_spectra=False, get_delta=False)

