import os
import time
from queue import Queue

import numpy as np
from matplotlib import pyplot as plt

import models
import transmit

from config import Config
from models import RgbDetector, SpecDetector
import cv2


def main():
    spec_detector = SpecDetector(blk_model_path=Config.blk_model_path, pixel_model_path=Config.pixel_model_path)
    rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                               background_model_path=Config.rgb_background_model_path)
    total_len = Config.nRows * Config.nCols * Config.nBands * 4  # float型变量, 4个字节
    total_rgb = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)
    if not os.access(rgb_fifo_path, os.F_OK):
        os.mkfifo(rgb_fifo_path, 0o777)
    while True:
        fd_img = os.open(img_fifo_path, os.O_RDONLY)
        fd_rgb = os.open(rgb_fifo_path, os.O_RDONLY)

        # spec data read
        data = os.read(fd_img, total_len)
        if len(data) < 3:
            threshold = int(float(data))
            Config.spec_size_threshold = threshold
            print("[INFO] Get spec threshold: ", threshold)
        else:
            data_total = data
        os.close(fd_img)
        
        # rgb data read
        rgb_data = os.read(fd_rgb, total_rgb)
        if len(rgb_data) < 3:
            rgb_threshold = int(float(rgb_data))
            Config.rgb_size_threshold = rgb_threshold
            print("[INFO] Get rgb threshold", rgb_threshold)
            continue
        else:
            rgb_data_total = rgb_data
        os.close(fd_rgb)

        # 识别
        t1 = time.time()
        img_data = np.frombuffer(data_total, dtype=np.float32).reshape((Config.nRows, Config.nBands, -1)) \
            .transpose(0, 2, 1)
        rgb_data = np.frombuffer(rgb_data_total, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
        # 光谱识别
        mask = spec_detector.predict(img_data)
        # rgb识别
        mask_rgb = rgb_detector.predict(rgb_data)
        # 结果合并
        mask_result = (mask | mask_rgb).astype(np.uint8)
        # mask_result = mask_rgb.astype(np.uint8)
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        t2 = time.time()
        print(f'rgb len = {len(rgb_data)}')

        # 写出
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask_result.tobytes())
        os.close(fd_mask)
        t3 = time.time()
        print(f'total time is:{t3 - t1}')


def read_c_captures(buffer_path, no_mask=True, nrows=256, ncols=1024, selected_bands=None):
    if os.path.isdir(buffer_path):
        buffer_names = [buffer_name for buffer_name in os.listdir(buffer_path) if buffer_name.endswith('.raw')]
    else:
        buffer_names = [buffer_path, ]
    for buffer_name in buffer_names:
        with open(os.path.join(buffer_path, buffer_name), 'rb') as f:
            data = f.read()
        img = np.frombuffer(data, dtype=np.float32).reshape((nrows, -1, ncols)) \
            .transpose(0, 2, 1)
        if selected_bands is not None:
            img = img[..., selected_bands]
        if img.shape[0] == 1:
            img = img[0, ...]
        if not no_mask:
            mask_name = buffer_name.replace('buf', 'mask').replace('.raw', '')
            with open(os.path.join(buffer_path, mask_name), 'rb') as f:
                data = f.read()
            mask = np.frombuffer(data, dtype=np.uint8).reshape((nrows, ncols, -1))
        else:
            mask_name = "no mask"
            mask = np.zeros_like(img)
        # mask = cv2.resize(mask, (1024, 256))
        fig, axs = plt.subplots(2, 1)
        axs[0].matshow(img)
        axs[0].set_title(buffer_name)
        axs[1].imshow(mask)
        axs[1].set_title(mask_name)
        plt.show()


if __name__ == '__main__':
    # 相关参数
    img_fifo_path = "/tmp/dkimg.fifo"
    mask_fifo_path = "/tmp/dkmask.fifo"
    rgb_fifo_path = "/tmp/dkrgb.fifo"

    # 主函数
    main()
    # read_c_captures('/home/lzy/2022.7.15/tobacco_v1_0/', no_mask=True, nrows=256, ncols=1024,
    #                 selected_bands=[380, 300, 200])
