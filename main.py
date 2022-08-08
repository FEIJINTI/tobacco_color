import os
import sys

import cv2
import time
import numpy as np

import utils
from config import Config
from models import RgbDetector, SpecDetector
import logging


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
    logging.info(f"请注意!正在以调试模式运行程序，输出的信息可能较多。")
    while True:
        fd_img = os.open(img_fifo_path, os.O_RDONLY)
        fd_rgb = os.open(rgb_fifo_path, os.O_RDONLY)

        # spec data read
        data = os.read(fd_img, total_len)
        if len(data) < 3:
            try:
                threshold = int(float(data))
                Config.spec_size_threshold = threshold
                logging.info('[INFO] Get spec threshold: ', threshold)
            except Exception as e:
                logging.error(f'毁灭性错误:收到长度小于3却无法转化为整数spec_size_threshold的网络报文，报文内容为 {data},'
                              f' 错误为 {e}.')
        else:
            data_total = data
        os.close(fd_img)
        # rgb data read
        rgb_data = os.read(fd_rgb, total_rgb)
        if len(rgb_data) < 3:
            try:
                rgb_threshold = int(float(rgb_data))
                Config.rgb_size_threshold = rgb_threshold
                logging.info(f'Get rgb threshold: {rgb_threshold}')
            except Exception as e:
                logging.error(f'毁灭性错误:收到长度小于3却无法转化为整数spec_size_threshold的网络报文，报文内容为 {total_rgb},'
                              f' 错误为 {e}.')
            continue
        else:
            rgb_data_total = rgb_data
        os.close(fd_rgb)
        # 识别
        since = time.time()
        try:
            img_data = np.frombuffer(data_total, dtype=np.float32).reshape((Config.nRows, Config.nBands, -1)) \
            .transpose(0, 2, 1)
        except Exception as e:
            logging.error(f'毁灭性错误!收到的光谱数据长度为{len(data_total)}无法转化成指定的形状 {e}')
        try:
            rgb_data = np.frombuffer(rgb_data_total, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
        except Exception as e:
            logging.error(f'毁灭性错误!收到的rgb数据长度为{len(rgb_data)}无法转化成指定形状 {e}')
        if only_spec:
            # 光谱识别
            mask_spec = spec_detector.predict(img_data).astype(np.uint8)
            _ = rgb_detector.predict(rgb_data)
            mask_rgb = np.zeros_like(mask_spec, dtype=np.uint8)
        elif only_color:
            # rgb识别
            _ = spec_detector.predict(img_data)
            mask_rgb = rgb_detector.predict(rgb_data).astype(np.uint8)
            mask_spec = np.zeros_like(mask_rgb, dtype=np.uint8)
        else:
            mask_spec = spec_detector.predict(img_data).astype(np.uint8)
            mask_rgb = rgb_detector.predict(rgb_data).astype(np.uint8)
        # 进行多个喷阀的合并
        masks = [utils.valve_expend(mask) for mask in [mask_spec, mask_rgb]]
        # 进行喷阀同时开启限制
        masks = [utils.valve_limit(mask, Config.max_open_valve_limit) for mask in masks]
        # control the size of the output masks, 在resize前，图像的宽度是和喷阀对应的
        masks = [cv2.resize(mask.astype(np.uint8), Config.target_size) for mask in masks]
        # 写出
        output_fifos = [mask_fifo_path, rgb_mask_fifo_path]
        for fifo, mask in zip(output_fifos, masks):
            fd_mask = os.open(fifo, os.O_WRONLY)
            os.write(fd_mask, mask.tobytes())
            os.close(fd_mask)
        time_spent = (time.time() - since) * 1000
        logging.info(f'Total time is: {time_spent:.2f} ms')
        if time_spent > 200:
            logging.warning(f'警告预测超时,预测耗时超过了200ms,The prediction time is {time_spent:.2f} ms.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='主程序')
    parser.add_argument('-oc', default=False, action='store_true', help='只进行RGB彩色预测 only rgb', required=False)
    parser.add_argument('-os', default=False, action='store_true', help='只进行光谱预测 only spec', required=False)
    parser.add_argument('-d', default=False, action='store_true', help='是否使用DEBUG模式', required=False)
    args = parser.parse_args()
    # fifo 参数
    img_fifo_path = '/tmp/dkimg.fifo'
    rgb_fifo_path = '/tmp/dkrgb.fifo'
    # mask fifo
    mask_fifo_path = '/tmp/dkmask.fifo'
    rgb_mask_fifo_path = '/tmp/dkmask_rgb.fifo'
    # logging相关
    file_handler = logging.FileHandler(os.path.join(Config.root_dir, '.tobacco_algorithm.log'))
    file_handler.setLevel(logging.DEBUG if args.d else logging.WARNING)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler], level=logging.DEBUG)
    main(only_spec=args.os, only_color=args.oc)
