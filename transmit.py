import os
import threading
from queue import Queue
import numpy as np
from config import Config


class Receiver(object):
    def __init__(self):
        self.output = None

    def set_source(self, *args, **kwargs):
        """
        用于设置数据的来源，每个receiver仅允许有单个来源
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def set_output(self, output: Queue):
        """
        设置单个输出源
        :param output:
        :return:
        """
        self.output = output

    def start(self, *args, **kwargs):
        """
        启动接收线程或进程
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def stop(self, *args, **kwargs):
        """
        停止接收线程或进程
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class FifoReceiver(Receiver):
    def __init__(self, fifo_path: str, output: Queue, read_max_num: int):
        super().__init__()
        self._input_fifo_path = None
        self._output_queue = None
        self._max_len = read_max_num

        self.set_source(fifo_path)
        self.set_output(output)
        self._need_stop = threading.Event()
        self._need_stop.clear()

    def set_source(self, fifo_path: str):
        if not os.access(fifo_path, os.F_OK):
            os.mkfifo(fifo_path, 0o777)
        self._input_fifo_path = fifo_path

    def set_output(self, output: Queue):
        self._output_queue = output

    def start(self, post_process_func=None, name='fifo_receiver'):
        x = threading.Thread(target=self._receive_thread_func,
                             name=name, args=(post_process_func, ))
        x.start()

    def stop(self):
        self._need_stop.set()

    def _receive_thread_func(self, post_process_func=None):
        """
        接收线程

        :param post_process_func:
        :return:
        """
        while not self._need_stop.is_set():
            input_fifo = os.open(self._input_fifo_path, os.O_RDONLY)
            data = os.read(input_fifo, self._max_len)
            if post_process_func is not None:
                data = post_process_func(data)
            self._output_queue.put(data)
            os.close(input_fifo)
        self._need_stop.clear()

    @staticmethod
    def spec_data_post_process(data):
        if len(data) < 3:
            threshold = int(float(data))
            print("[INFO] Get Spec threshold: ", threshold)
            return threshold
        else:
            spec_img = np.frombuffer(data, dtype=np.float32).\
                reshape((Config.nRows, Config.nBands, -1)).transpose(0, 2, 1)
            return spec_img

    @staticmethod
    def rgb_data_post_process(data):
        if len(data) < 3:
            threshold = int(float(data))
            print("[INFO] Get RGB threshold: ", threshold)
            return threshold
        else:
            rgb_img = np.frombuffer(data, dtype=np.uint8).reshape((Config.nRgbRows, Config.nRgbCols, -1))
            return rgb_img
