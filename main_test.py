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
        delta = (delta[0] - search_area_size[1]//2, delta[1] - search_area_size[1]//2)
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


class ValveTest:
    def __init__(self):
        self.reminder = """
        快，给我个指令：
        a. 开始命令 st                                       e. 设置 光谱(a)相机 的延时，格式 e,500
        b. 停止命令 sp                                       f. 设置 彩色(b)相机 的延时, 格式 f,500
        c. 设置光谱相机分频, 得是4的倍数而且>=8, 格式: c,8       g. 发个da和db完全重叠的mask
        d. 阀板的脉冲分频系数, >=2即可                         h. 发个da和db呈现出X形的mask
        或者你给我个小于256的数字，我就测试对应的喷阀，按q键我就退出。\n
        """
        self.s = socket.socket()  # 创建 socket 对象
        host = socket.gethostname()  # 获取本地主机名
        port = 13452  # 设置端口
        self.s.bind((host, port))  # 绑定端口
        self.s.listen(5)  # 等待客户端连接

    def run(self):
        while True:
            logging.info("我在等连接...")
            c, addr = self.s.accept()  # 建立客户端连接
            print('Connection Established：', addr)
            value = input(self.reminder)
            if value == 'q':
                break
            else:
                self.process_cmd(value)
        c.close()  # 关闭连接

    def pad_cmd(self, cmd):
        return b'\xAA'+cmd+b'\xFF\xFF\xBB'

    def param_cmd_parser(self, cmd, default_value, checker=None):
        try:
            value = int(cmd.split(',')[-1])
        except:
            print(f'你给的值不对啊，我先给你弄个{default_value}吧')
            value = default_value
        if checker is not None:
            if not checker(value):
                return None
        return value

    def process_cmd(self, value):
        if value == 'a':
            # a.开始命令
            cmd = b'\x00\x03' + 'sa'.encode('ascii') + b'\xFF'
        elif value == 'b':
            # b.停止命令
            cmd = b'\x00\x03' + 'sb'.encode('ascii') + b'\xFF'
        elif value.startswith('c'):
            # c. 设置光谱相机分频，得是4的倍数而且>=8，格式：c,8
            checker = lambda x: (x // 4 == 0) and (x >= 8)
            value = self.param_cmd_parser(value, default_value=8, checker=checker)
            if value is None:
                print("值需要是4的倍数且大于8")
                return
            cmd = b'\x00\x0a' + 'sc'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('d'):
            # d. 阀板的脉冲分频系数，>=2即可
            checker = lambda x: x >= 2
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于2")
                return
            cmd = b'\x00\x0a' + 'sv'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('e'):
            # e. 设置 光谱(a)相机 的延时，格式 e,500
            checker = lambda x: (x >= 0)
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于0")
                return
            cmd = b'\x00\x0a' + 'sa'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value.startswith('f'):
            # f. 设置 RGB(b)相机 的延时，格式 e,500
            checker = lambda x: (x >= 0)
            value = self.param_cmd_parser(value, default_value=2, checker=checker)
            if value is None:
                print("你得大于等于0")
                return
            cmd = b'\x00\x0a' + 'sb'.encode('ascii') + f"{value:08d}".encode('ascii')
        elif value == 'g':
            # g.发个da和db完全重叠的mask
            mask_a, mask_b = np.eye(256, dtype=np.uint8), np.eye(256, dtype=np.uint8)
            len_a, data_a = self.format_data(mask_a)
            len_b, data_b = self.format_data(mask_b)
            cmd = len_a + 'da'.encode('ascii') + mask_a
            self.send(cmd)
            cmd = len_b + 'db'.encode('ascii') + mask_b
        elif value == 'h':
            # h.发个da和db呈现出X形的mask
            mask_a, mask_b = np.eye(256, dtype=np.uint8), np.eye(256, dtype=np.uint8).T
            len_a, data_a = self.format_data(mask_a)
            len_b, data_b = self.format_data(mask_b)
            cmd = len_a + 'da'.encode('ascii') + mask_a
            self.send(cmd)
            cmd = len_b + 'db'.encode('ascii') + mask_b
        else:
            try:
                value = int(value)
            except Exception as e:
                print(e)
                print(f"你给的指令: {value} 咋看都不对")
                return
        self.send(cmd)

    def send(self, cmd: bytes) -> None:
        print("我要send 这个了:\n")
        print(cmd)
        cmd = self.pad_cmd(cmd)
        try:
            self.s.send(cmd)
        except Exception as e:
            print(f"发失败了, 这是我找到的错误信息\n{e}")
            return
        print("发好了")

    def format_data(self, array_to_send: np.ndarray) -> (bytes, bytes):
        data = np.packbits(array_to_send, axis=-1)
        data = data.tobytes()
        data_len = (len(data)+2).to_bytes(2, 'big')
        return data_len, data


if __name__ == '__main__':
    testor = TestMain()
    testor.pony_run(test_path=r'/home/lzy/2022.7.30/tobacco_v1_0/saved_img/',
                    test_rgb=True, test_spectra=True, get_delta=False)
