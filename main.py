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
    if not os.access(rgb_fifo_path, os.F_OK):
        os.mkfifo(rgb_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)

    # 进行补偿buffer的开启
    if Config.offset_vertical < 0:
        # 纵向的补偿小于0，那就意味着光谱图要上移才能补上，那么我们应该补偿SPEC相机的全 0 图像
        conserve_part = np.zeros((abs(Config.offset_vertical) // 4, Config.nRows, Config.nBands))
    elif Config.offset_vertical > 0:
        # 纵向的补偿小于0，说明光谱图下移才能补上去，那么我们就需要补偿RGB相机的全 0 图像
        conserve_part = np.zeros(abs(Config.offset_vertical), Config.nRgbRows, Config.nRgbBands)
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
        img_data = np.frombuffer(data_total, dtype=np.float32).reshape((Config.nRows, Config.nBands, -1))\
                     .transpose(0, 2, 1)
        rgb_data = np.frombuffer(rgb_data_total, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
        if Config.offset_vertical < 0:
            # 纵向的补偿小于0，那就意味着光谱图要上移才能补上，那么我们应该补偿SPEC相机的全 0 图像
            new_conserve_part, real_part = img_data[:abs(Config.offset_vertical) // 4, ...],\
                                       img_data[abs(Config.offset_vertical) // 4:, ...]
            img_data = np.concatenate([real_part, conserve_part], axis=0)
            conserve_part = new_conserve_part
        elif Config.offset_vertical > 0:
            # 纵向的补偿小于0，说明光谱图下移才能补上去，那么我们就需要补偿RGB相机的全 0 图像
            new_conserve_part, real_part = rgb_data[:abs(Config.offset_vertical), ...],\
                                           rgb_data[abs(Config.offset_vertical):, ...]
            rgb_data = np.concatenate([real_part, conserve_part], axis=0)
            conserve_part = new_conserve_part
        # 光谱识别
        mask_spec = spec_detector.predict(img_data)
        # rgb识别
        mask_rgb = rgb_detector.predict(rgb_data)
        # 结果合并
        mask_result = (mask_spec | mask_rgb).astype(np.uint8)

        # control the size of the output masks
        masks = [cv2.resize(mask.astype(np.uint8), Config.target_size) for mask in [mask_spec, mask_rgb]]
        # 写出
        output_fifos = [mask_fifo_path, ]
        for fifo, mask in zip(output_fifos, masks):
            fd_mask = os.open(fifo, os.O_WRONLY)
            os.write(fd_mask, mask.tobytes())
            os.close(fd_mask)
        t3 = time.time()
        print(f'total time is:{t3 - t1}')


if __name__ == '__main__':
    # 相关参数
    img_fifo_path = "/tmp/dkimg.fifo"
    rgb_fifo_path = "/tmp/dkrgb.fifo"

    mask_fifo_path = "/tmp/dkmask.fifo"
    # 主函数
    main()
    # read_c_captures('/home/lzy/2022.7.15/tobacco_v1_0/', no_mask=True, nrows=256, ncols=1024,
    #                 selected_bands=[380, 300, 200])
