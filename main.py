import os
from datetime import datetime
from pathlib import Path
import sys

import cv2
import time
import numpy as np

import utils
import utils as utils_customized
from config import Config
from models import RgbDetector, SpecDetector
import logging


def main(only_spec=False, only_color=False, if_merge=False, interval_time=None, delay_repeat_time=None,
         single_spec=False, single_color=False):
    spec_detector = SpecDetector(blk_model_path=Config.blk_model_path, pixel_model_path=Config.pixel_model_path)
    rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                               background_model_path=Config.rgb_background_model_path,
                               ai_path=Config.ai_path)
    _, _ = spec_detector.predict(np.ones((Config.nRows, Config.nCols, Config.nBands), dtype=float) * 0.4), \
           rgb_detector.predict(np.ones((Config.nRgbRows, Config.nRgbCols, Config.nRgbBands), dtype=np.uint8) * 40)
    total_len = Config.nRows * Config.nCols * Config.nBands * 4  # float型变量, 4个字节
    total_rgb = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
    log_file_name = datetime.now().strftime('%Y_%m_%d__%H_%M_%S.log')
    if single_spec:
        os.makedirs(Path(Config.root_dir) / Path(Config.rgb_log_dir), exist_ok=True)
        log_path = Path(Config.root_dir) / Path(Config.rgb_log_dir) / log_file_name
    if single_color:
        os.makedirs(Path(Config.root_dir) / Path(Config.spec_log_dir), exist_ok=True)
        log_path = Path(Config.root_dir) / Path(Config.spec_log_dir) / log_file_name
    if not single_color:
        logging.info("create color fifo")
        if not os.access(img_fifo_path, os.F_OK):
            os.mkfifo(img_fifo_path, 0o777)
        if not os.access(mask_fifo_path, os.F_OK):
            os.mkfifo(mask_fifo_path, 0o777)
    if not single_spec:
        logging.info("create rgb fifo")
        if not os.access(rgb_fifo_path, os.F_OK):
            os.mkfifo(rgb_fifo_path, 0o777)
        if not os.access(rgb_mask_fifo_path, os.F_OK):
            os.mkfifo(rgb_mask_fifo_path, 0o777)
    logging.info(f"请注意!正在以调试模式运行程序，输出的信息可能较多。")
    # specially designed for Miaow.
    if (interval_time is not None) and (delay_repeat_time is not None):
        interval_time = float(interval_time) / 1000.0
        delay_repeat_time = int(delay_repeat_time)
        logging.warning(f'Delay {interval_time * 1000:.2f}ms will be added per {delay_repeat_time} frames')
        delay_repeat_time_count = 0

    log_time_count, value_num_count = 0, 0
    while True:
        img_data, rgb_data = None, None
        if single_spec:
            fd_img = os.open(img_fifo_path, os.O_RDONLY)
            # spec data read
            data_total = os.read(fd_img, total_len)
            if len(data_total) < 3:
                try:
                    threshold = int(float(data_total))
                    Config.spec_size_threshold = threshold
                    logging.info(f'[INFO] Get spec threshold: {threshold}')
                except Exception as e:
                    logging.error(
                        f'毁灭性错误:收到长度小于3却无法转化为整数spec_size_threshold的网络报文，报文内容为 {data_total},'
                        f' 错误为 {e}.')
                if single_spec:
                    continue
            else:
                data_total = data_total
            os.close(fd_img)
            try:
                img_data = np.frombuffer(data_total, dtype=np.float32).reshape((Config.nRows, Config.nBands, -1)) \
                    .transpose(0, 2, 1)
            except Exception as e:
                logging.error(f'毁灭性错误!收到的光谱数据长度为{len(data_total)}无法转化成指定的形状 {e}')

        if single_color:
            fd_rgb = os.open(rgb_fifo_path, os.O_RDONLY)
            # rgb data read
            rgb_data_total = os.read(fd_rgb, total_rgb)
            if len(rgb_data_total) < 3:
                try:
                    rgb_threshold = int(float(rgb_data_total))
                    Config.rgb_size_threshold = rgb_threshold
                    logging.info(f'Get rgb threshold: {rgb_threshold}')
                except Exception as e:
                    logging.error(
                        f'毁灭性错误:收到长度小于3却无法转化为整数spec_size_threshold的网络报文，报文内容为 {total_rgb},'
                        f' 错误为 {e}.')
                continue
            else:
                rgb_data_total = rgb_data_total
            os.close(fd_rgb)
            try:
                rgb_data = np.frombuffer(rgb_data_total, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
            except Exception as e:
                logging.error(f'毁灭性错误!收到的rgb数据长度为{len(rgb_data_total)}无法转化成指定形状 {e}')

        # 识别 read
        since = time.time()
        # predict
        if single_spec or single_color:
            if single_spec:
                mask_spec = spec_detector.predict(img_data).astype(np.uint8)
                masks = [mask_spec, ]
            else:
                mask_rgb = rgb_detector.predict(rgb_data).astype(np.uint8)
                masks = [mask_rgb, ]
        else:
            if only_spec:
                # 光谱识别
                mask_spec = spec_detector.predict(img_data).astype(np.uint8)
                mask_rgb = np.zeros_like(mask_spec, dtype=np.uint8)
            elif only_color:
                # rgb识别
                mask_rgb = rgb_detector.predict(rgb_data).astype(np.uint8)
                mask_spec = np.zeros_like(mask_rgb, dtype=np.uint8)
            else:
                mask_spec = spec_detector.predict(img_data).astype(np.uint8)
                mask_rgb = rgb_detector.predict(rgb_data).astype(np.uint8)
            masks = [mask_spec, mask_rgb]
        # 进行多个喷阀的合并
        masks = [utils_customized.shield_valve(mask, left_shield=10, right_shield=10) for mask in masks]
        masks = [utils_customized.valve_expend(mask) for mask in masks]
        mask_nums = sum([np.sum(np.sum(mask)) for mask in masks])
        log_time_count += 1
        value_num_count += mask_nums
        if log_time_count > Config.log_freq:
            utils.valve_log(log_path, valve_num=value_num_count)
            log_time_count, value_num_count = 0, 0

        # 进行喷阀同时开启限制,在8月11日后收到倪超老师的电话，关闭
        # masks = [utils_customized.valve_limit(mask, Config.max_open_valve_limit) for mask in masks]
        # control the size of the output masks, 在resize前，图像的宽度是和喷阀对应的
        masks = [cv2.resize(mask.astype(np.uint8), Config.target_size) for mask in masks]
        # merge the masks if needed
        if if_merge and (len(masks) > 1):
            masks = [masks[0] | masks[1], masks[1]]
        if (interval_time is not None) and (delay_repeat_time is not None):
            delay_repeat_time_count += 1
            if delay_repeat_time_count > delay_repeat_time:
                logging.warning(f"Delay time {interval_time * 1000:.2f}ms after {delay_repeat_time} frames")
                delay_repeat_time_count = 0
                time.sleep(interval_time)
        # 写出
        if single_spec:
            output_fifos = [mask_fifo_path, ]
        elif single_color:
            output_fifos = [rgb_mask_fifo_path, ]
        else:
            output_fifos = [mask_fifo_path, rgb_mask_fifo_path]
        for fifo, mask in zip(output_fifos, masks):
            fd_mask = os.open(fifo, os.O_WRONLY)
            os.write(fd_mask, mask.tobytes())
            os.close(fd_mask)
        time_spent = (time.time() - since) * 1000
        predict_by = 'spec' if single_spec else 'rgb' if single_color else 'spec+rgb'
        logging.info(f'Total time is: {time_spent:.2f} ms, predicted by {predict_by}')
        if time_spent > Config.max_time_spent:
            logging.warning(f'警告预测超时,预测耗时超过了200ms,The prediction time is {time_spent:.2f} ms.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='主程序')
    parser.add_argument('-oc', default=False, action='store_true', help='只进行RGB彩色预测 only rgb', required=False)
    parser.add_argument('-os', default=False, action='store_true', help='只进行光谱预测 only spec', required=False)
    parser.add_argument('-sc', default=False, action='store_true', help='只进行RGB预测且只返回一个mask', required=False)
    parser.add_argument('-ss', default=False, action='store_true', help='只进行光谱预测且只返回一个mask',
                        required=False)
    parser.add_argument('-m', default=False, action='store_true', help='if merge the two masks', required=False)
    parser.add_argument('-d', default=False, action='store_true', help='是否使用DEBUG模式', required=False)
    parser.add_argument('-dt', default=None, help='delay time', required=False)
    parser.add_argument('-df', default=None, help='delay occours after how many frames', required=False)
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
    console_handler.setLevel(logging.DEBUG if args.d else logging.WARNING)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[file_handler, console_handler], level=logging.DEBUG)
    main(only_spec=args.os, only_color=args.oc, if_merge=args.m, interval_time=args.dt, delay_repeat_time=args.df,
         single_spec=args.ss, single_color=args.sc)
