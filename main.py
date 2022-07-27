import os
import time
import numpy as np
import scipy.io

from config import Config
from models import RgbDetector, SpecDetector, ManualTree, AnonymousColorDetector
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
        data = os.read(fd_img, total_len)
        # 读取(开启一个管道)
        if len(data) < 3:
            threshold = int(float(data))
            Config.spec_size_threshold = threshold
            print("[INFO] Get threshold: ", threshold)
            continue
        else:
            data_total = data
            rgb_data = os.read(fd_rgb, total_rgb)
            if len(rgb_data) < 3:
                rgb_threshold = int(float(rgb_data))
                Config.rgb_size_threshold = rgb_threshold
                print(rgb_threshold)
                continue
            else:
                rgb_data_total = rgb_data
        os.close(fd_img)
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
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        t2 = time.time()
        print(f'rgb len = {len(rgb_data)}')

        # 写出
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask_result.tobytes())
        os.close(fd_mask)
        t3 = time.time()
        print(f'total time is:{t3 - t1}')


def save_main():
    threshold = Config.spec_size_threshold
    rgb_threshold = Config.rgb_size_threshold
    manual_tree = ManualTree(blk_model_path=Config.blk_model_path, pixel_model_path=Config.pixel_model_path)
    tobacco_detector = AnonymousColorDetector(file_path=Config.rgb_tobacco_model_path)
    background_detector = AnonymousColorDetector(file_path=Config.rgb_background_model_path)
    total_len = Config.nRows * Config.nCols * Config.nBands * 4  # float型变量, 4个字节
    total_rgb = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)
    if not os.access(rgb_fifo_path, os.F_OK):
        os.mkfifo(rgb_fifo_path, 0o777)
    img_list = []
    idx = 0
    while idx <= 30:
        idx += 1
        fd_img = os.open(img_fifo_path, os.O_RDONLY)
        fd_rgb = os.open(rgb_fifo_path, os.O_RDONLY)
        data = os.read(fd_img, total_len)

        # 读取(开启一个管道)
        if len(data) < 3:
            threshold = int(float(data))
            print("[INFO] Get threshold: ", threshold)
            continue
        else:
            data_total = data
            rgb_data = os.read(fd_rgb, total_rgb)
            if len(rgb_data) < 3:
                rgb_threshold = int(float(rgb_data))
                print(rgb_threshold)
                continue
            else:
                rgb_data_total = rgb_data
        os.close(fd_img)
        os.close(fd_rgb)

        # 识别
        t1 = time.time()
        img_data = np.frombuffer(data_total, dtype=np.float32).reshape((Config.nRows, Config.nBands, -1)). \
            transpose(0, 2, 1)
        rgb_data = np.frombuffer(rgb_data_total, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
        img_list.append((rgb_data.copy(), img_data.copy()))

        pixel_predict_result = manual_tree.pixel_predict_ml_dilation(data=img_data, iteration=1)
        blk_predict_result = manual_tree.blk_predict(data=img_data)
        rgb_data = tobacco_detector.pretreatment(rgb_data)
        # print(rgb_data.shape)
        rgb_predict_result = 1 - (background_detector.predict(rgb_data, threshold_low=Config.threshold_low,
                                                              threshold_high=Config.threshold_high) |
                                  tobacco_detector.swell(tobacco_detector.predict(rgb_data,
                                                                                  threshold_low=Config.threshold_low,
                                                                                  threshold_high=Config.threshold_high)))
        mask_rgb = rgb_predict_result.reshape(Config.nRows, Config.nCols // Config.blk_size, Config.blk_size) \
            .sum(axis=2).reshape(Config.nRows // 4, Config.blk_size, Config.nCols // Config.blk_size) \
            .sum(axis=1)
        mask_rgb[mask_rgb <= rgb_threshold] = 0
        mask_rgb[mask_rgb > rgb_threshold] = 1
        mask = (pixel_predict_result & blk_predict_result).astype(np.uint8)
        mask = mask.reshape(Config.nRows, Config.nCols // Config.blk_size, Config.blk_size) \
            .sum(axis=2).reshape(Config.nRows // 4, Config.blk_size, Config.nCols // Config.blk_size) \
            .sum(axis=1)
        mask[mask <= threshold] = 0
        mask[mask > threshold] = 1
        # mask_result = (mask | mask_rgb).astype(np.uint8)
        mask_result = mask_rgb
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        t2 = time.time()
        print(f'rgb len = {len(rgb_data)}')

        # 写出
        fd_mask = os.open(mask_fifo_path, os.O_WRONLY)
        os.write(fd_mask, mask_result.tobytes())
        os.close(fd_mask)
        t3 = time.time()
        print(f'total time is:{t3 - t1}')
    i = 0
    print("Stop Serving")
    for img in img_list:
        print(f"writing img {i}...")
        cv2.imwrite(f"./{i}.png", img[0][..., ::-1])
        np.save(f'./{i}.npy', img[1])
        i += 1
    print("save success")


if __name__ == '__main__':
    # 相关参数
    img_fifo_path = "/tmp/dkimg.fifo"
    mask_fifo_path = "/tmp/dkmask.fifo"
    rgb_fifo_path = "/tmp/dkrgb.fifo"

    # 主函数
    main()
