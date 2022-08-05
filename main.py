import os
import cv2
import time
import numpy as np

from config import Config
from models import RgbDetector, SpecDetector


def main(only_spec=False, only_color=False):
    spec_detector = SpecDetector(blk_model_path=Config.blk_model_path, pixel_model_path=Config.pixel_model_path)
    rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                               background_model_path=Config.rgb_background_model_path)
    _, _ = spec_detector.predict(np.ones((Config.nRows, Config.nCols, Config.nBands), dtype=float)*0.4),\
           rgb_detector.predict(np.ones((Config.nRgbRows, Config.nRgbCols, Config.nRgbBands), dtype=np.uint8)*40)
    total_len = Config.nRows * Config.nCols * Config.nBands * 4  # float型变量, 4个字节
    total_rgb = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
    if not os.access(img_fifo_path, os.F_OK):
        os.mkfifo(img_fifo_path, 0o777)
    if not os.access(rgb_fifo_path, os.F_OK):
        os.mkfifo(rgb_fifo_path, 0o777)
    if not os.access(mask_fifo_path, os.F_OK):
        os.mkfifo(mask_fifo_path, 0o777)
    if not os.access(rgb_mask_fifo_path, os.F_OK):
        os.mkfifo(rgb_mask_fifo_path, 0o777)
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
        if only_spec:
            # 光谱识别
            mask_spec = spec_detector.predict(img_data)
            mask_rgb = np.zeros_like(mask_spec, dtype=np.uint8)
        elif only_color:
            # rgb识别
            mask_rgb = rgb_detector.predict(rgb_data)
            mask_spec = np.zeros_like(mask_rgb, dtype=np.uint8)
        else:
            mask_spec = spec_detector.predict(img_data)
            mask_rgb = rgb_detector.predict(rgb_data)

        # control the size of the output masks
        masks = [cv2.resize(mask.astype(np.uint8), Config.target_size) for mask in [mask_spec, mask_rgb]]
        # 写出
        output_fifos = [mask_fifo_path, rgb_mask_fifo_path]
        for fifo, mask in zip(output_fifos, masks):
            fd_mask = os.open(fifo, os.O_WRONLY)
            os.write(fd_mask, mask.tobytes())
            os.close(fd_mask)
        t3 = time.time()
        print(f'total time is:{t3 - t1}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='主程序')
    parser.add_argument('-oc', default=False, action='store_true', help='只进行RGB彩色预测 only rgb', required=False)
    parser.add_argument('-os', default=False, action='store_true', help='只进行光谱预测 only spec', required=False)
    args = parser.parse_args()
    # fifo 参数
    img_fifo_path = "/tmp/dkimg.fifo"
    rgb_fifo_path = "/tmp/dkrgb.fifo"
    # mask fifo
    mask_fifo_path = "/tmp/dkmask.fifo"
    rgb_mask_fifo_path = "/tmp/dkmask_rgb.fifo"
    main(only_spec=args.os, only_color=args.oc)
