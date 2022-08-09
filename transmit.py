import os
import threading
from multiprocessing import Process, Queue
import time

import cv2

import utils
from utils import ImgQueue as ImgQueue
import functools
import numpy as np
from config import Config
from models import SpecDetector, RgbDetector
from typing import Any
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    level=logging.WARNING)


class Transmitter(object):
    def __init__(self, job_name:str, run_process:bool = False):
        self.output = None
        self.job_name = job_name
        self.run_process = run_process # If true, run process when started else run thread.
        self._thread_stop = threading.Event()
        self._thread_stop.clear()
        self._running_handler = None

    def set_source(self, *args, **kwargs):
        """
        用于设置数据的来源，每个receiver仅允许有单个来源
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def set_output(self, output: ImgQueue):
        """
        设置单个输出源
        :param output:
        :return:
        """
        self.output = output

    def start(self, *args, **kwargs):
        """
        启动线程或进程
        :param args:
        :param kwargs:
        :return:
        """
        name = kwargs.get('name', default='base thread')
        if not self.run_process:
            self._running_handler = threading.Thread(target=self.job_func, name=name, args=args)
        else:
            self._running_handler = Process(target=self.job_func, name=name, args=args, daemon=True)
        self._running_handler.start()

    def stop(self, *args, **kwargs):
        """
        停止线程或进程
        :param args:
        :param kwargs:
        :return:
        """
        if self._running_handler is not None:
            self._thread_stop.set()
            self._running_handler = None

    @staticmethod
    def job_decorator(func):
        functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logging.info(f'{self.job_name} {"process" if self.run_process else "thread"} start.')
            while not self._thread_stop.is_set():
                self.job_func(*args, **kwargs)
            logging.info(f'{self.job_name} {"process" if self.run_process else "thread"} stop.')
            self._need_stop.clear()
        return wrapper

    def job_func(self, *args, **kwargs):
        raise NotImplementedError


class BeforeAfterMethods:
    @classmethod
    def mask_preprocess(cls, mask: np.ndarray):
        logging.info(f"Send mask with size {mask.shape}")
        return mask.tobytes()

    @classmethod
    def spec_data_post_process(cls, data):
        if len(data) < 3:
            threshold = int(float(data))
            logging.info(f"Get Spec threshold: {threshold}")
            return threshold
        else:
            spec_img = np.frombuffer(data, dtype=np.float32).\
                reshape((Config.nRows, Config.nBands, -1)).transpose(0, 2, 1)
            logging.info(f"Get SPEC image with size {spec_img.shape}")
            return spec_img

    @classmethod
    def rgb_data_post_process(cls, data):
        if len(data) < 3:
            threshold = int(float(data))
            logging.info(f"Get RGB threshold: {threshold}")
            return threshold
        else:
            rgb_img = np.frombuffer(data, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
            logging.info(f"Get RGB img with size {rgb_img.shape}")
            return rgb_img


class FifoReceiver(Transmitter):
    def __init__(self, job_name:str, fifo_path: str, output: ImgQueue,
                 read_max_num: int, msg_queue=None):
        super().__init__(job_name=job_name)
        self._input_fifo_path = None
        self._output_queue = None
        self._msg_queue = msg_queue
        self._max_len = read_max_num

        self.set_source(fifo_path)
        self.set_output(output)

    def set_source(self, fifo_path: str):
        if not os.access(fifo_path, os.F_OK):
            os.mkfifo(fifo_path, 0o777)
        self._input_fifo_path = fifo_path

    def set_output(self, output: ImgQueue):
        self._output_queue = output

    @Transmitter.job_decorator
    def job_func(self, post_process_func=None):
        """
        接收线程

        :param post_process_func:
        :return:
        """
        input_fifo = os.open(self._input_fifo_path, os.O_RDONLY)
        data = os.read(input_fifo, self._max_len)
        if post_process_func is not None:
            data = post_process_func(data)
        self._output_queue.safe_put(data)
        os.close(input_fifo)



class FifoSender(Transmitter):
    def __init__(self, output_fifo_path: str, source: ImgQueue):
        super().__init__()
        self._input_source = None
        self._output_fifo_path = None
        self.set_source(source)
        self.set_output(output_fifo_path)
        self._need_stop = threading.Event()
        self._need_stop.clear()
        self._running_thread = None

    def set_source(self, source: ImgQueue):
        self._input_source = source

    def set_output(self, output_fifo_path: str):
        if not os.access(output_fifo_path, os.F_OK):
            os.mkfifo(output_fifo_path, 0o777)
        self._output_fifo_path = output_fifo_path

    def start(self, pre_process=None, name='fifo_receiver'):
        self._running_thread = threading.Thread(target=self._send_thread_func, name=name,
                                                args=(pre_process, ))
        self._running_thread.start()

    def stop(self):
        self._need_stop.set()

    def _send_thread_func(self, pre_process=None):
        """
        接收线程

        :param pre_process:
        :return:
        """
        while not self._need_stop.is_set():
            if self._input_source.empty():
                continue
            data = self._input_source.get()
            if pre_process is not None:
                data = pre_process(data)
            output_fifo = os.open(self._output_fifo_path, os.O_WRONLY)
            os.write(output_fifo, data)
            os.close(output_fifo)
        self._need_stop.clear()

    def __del__(self):
        self.stop()
        if self._running_thread is not None:
            self._running_thread.join()


class CmdImgSplitMidware(Transmitter):
    """
    用于控制命令和图像的中间件
    """
    def __init__(self, subscribers: typing.Dict[str, ImgQueue], rgb_queue: ImgQueue, spec_queue: ImgQueue):
        super().__init__()
        self._rgb_queue = None
        self._spec_queue = None
        self._subscribers = None
        self._server_thread = None
        self.set_source(rgb_queue, spec_queue)
        self.set_output(subscribers)
        self.thread_stop = threading.Event()

    def set_source(self, rgb_queue: ImgQueue, spec_queue: ImgQueue):
        self._rgb_queue = rgb_queue
        self._spec_queue = spec_queue

    def set_output(self, output: typing.Dict[str, ImgQueue]):
        self._subscribers = output

    def start(self, name='CMD_thread'):
        self._server_thread = threading.Thread(target=self._cmd_control_service, name=name)
        self._server_thread.start()

    def stop(self):
        self.thread_stop.set()

    def _cmd_control_service(self):
        while not self.thread_stop.is_set():
            # 判断是否有数据，如果没有数据那么就等下次吧，如果有数据来，必须保证同时
            if self._rgb_queue.empty() or self._spec_queue.empty():
                continue
            rgb_data = self._rgb_queue.get()
            spec_data = self._spec_queue.get()
            if isinstance(rgb_data, int) and isinstance(spec_data, int):
                # 看是不是命令需要执行如果是命令，就执行
                Config.rgb_size_threshold = rgb_data
                Config.spec_size_threshold = spec_data
                logging.info("获取到指令")
                continue
            elif isinstance(spec_data, np.ndarray) and isinstance(rgb_data, np.ndarray):
                # 如果是图片，交给预测的人
                for name, subscriber in self._subscribers.items():
                    item = (spec_data, rgb_data)
                    subscriber.safe_put(item)
            else:
                # 否则程序出现毁灭性问题，立刻崩
                logging.critical('两个相机传回的数据没有对上')
                raise Exception("两个相机传回的数据没有对上")
        self.thread_stop.clear()


class ImageSaver(Transmitter):
    """
    进行图片存储的中间件
    """
    def set_source(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass


class ThreadDetector(Transmitter):
    def __init__(self, input_queue: ImgQueue, output_queue: ImgQueue):
        super().__init__()
        self._input_queue, self._output_queue = input_queue, output_queue
        self._spec_detector = SpecDetector(blk_model_path=Config.blk_model_path,
                                           pixel_model_path=Config.pixel_model_path)
        self._rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                                         background_model_path=Config.rgb_background_model_path)
        self._predict_thread = None
        self._thread_exit = threading.Event()

    def set_source(self, img_queue: ImgQueue):
        self._input_queue = img_queue

    def stop(self, *args, **kwargs):
        self._thread_exit.set()

    def start(self, name='predict_thread'):
        self._predict_thread = threading.Thread(target=self._predict_server, name=name)
        self._predict_thread.start()

    def predict(self, spec: np.ndarray, rgb: np.ndarray):
        logging.info(f'Detector get image with shape {spec.shape} and {rgb.shape}')
        t1 = time.time()
        mask = self._spec_detector.predict(spec)
        t2 = time.time()
        logging.info(f'Detector finish spec predict within {(t2 - t1) * 1000:.2f}ms')
        # rgb识别
        mask_rgb = self._rgb_detector.predict(rgb)
        t3 = time.time()
        logging.info(f'Detector finish rgb predict within {(t3 - t2) * 1000:.2f}ms')
        # 结果合并
        mask_result = (mask | mask_rgb).astype(np.uint8)
        mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        t4 = time.time()
        logging.info(f'Detector finish merge within {(t4 - t3) * 1000: .2f}ms')
        logging.info(f'Detector finish predict within {(time.time() -t1)*1000:.2f}ms')
        return mask_result

    def _predict_server(self):
        while not self._thread_exit.is_set():
            if not self._input_queue.empty():
                spec, rgb = self._input_queue.get()
                mask = self.predict(spec, rgb)
                self._output_queue.safe_put(mask)
        self._thread_exit.clear()


class ProcessDetector(Transmitter):
    def __init__(self, input_queue: ImgQueue, output_queue: ImgQueue):
        super().__init__()
        self._input_queue, self._output_queue = input_queue, output_queue
        self._spec_detector = SpecDetector(blk_model_path=Config.blk_model_path,
                                           pixel_model_path=Config.pixel_model_path)
        self._rgb_detector = RgbDetector(tobacco_model_path=Config.rgb_tobacco_model_path,
                                         background_model_path=Config.rgb_background_model_path)
        self._predict_thread = None
        self._thread_exit = threading.Event()

    def set_source(self, img_queue: ImgQueue):
        self._input_queue = img_queue

    def stop(self, *args, **kwargs):
        self._thread_exit.set()

    def start(self, name='predict_thread'):
        self._predict_thread = Process(target=self._predict_server, name=name, daemon=True)
        self._predict_thread.start()

    def predict(self, spec: np.ndarray, rgb: np.ndarray):
        logging.debug(f'Detector get image with shape {spec.shape} and {rgb.shape}')
        t1 = time.time()
        mask_spec = self._spec_detector.predict(spec)
        t2 = time.time()
        logging.debug(f'Detector finish spec predict within {(t2 - t1) * 1000:.2f}ms')
        # rgb识别
        mask_rgb = self._rgb_detector.predict(rgb)
        t3 = time.time()
        logging.debug(f'Detector finish rgb predict within {(t3 - t2) * 1000:.2f}ms')
        # 结果合并
        # mask_result = (mask | mask_rgb).astype(np.uint8)
        # mask_result = mask_result.repeat(Config.blk_size, axis=0).repeat(Config.blk_size, axis=1).astype(np.uint8)
        # 进行多个喷阀的合并
        masks = [utils.valve_expend(mask) for mask in [mask_spec, mask_rgb]]
        # 进行喷阀同时开启限制
        masks = [utils.valve_limit(mask, Config.max_open_valve_limit) for mask in masks]
        # control the size of the output masks, 在resize前，图像的宽度是和喷阀对应的
        masks = [cv2.resize(mask.astype(np.uint8), Config.target_size) for mask in masks]
        t4 = time.time()
        logging.debug(f'Detector finish merge within {(t4 - t3) * 1000: .2f}ms')
        logging.debug(f'Detector finish predict within {(time.time() -t1)*1000:.2f}ms')
        return masks

    def _predict_server(self):
        while not self._thread_exit.is_set():
            if not self._input_queue.empty():
                spec, rgb = self._input_queue.get()
                masks = self.predict(spec, rgb)
                self._output_queue.put(masks[:])
        self._thread_exit.clear()


class SplitMidware(Transmitter):
    def set_source(self, mask_source: ImgQueue):


    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass