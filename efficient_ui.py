import cv2
import numpy as np

import transmit
from config import Config

from utils import ImgQueue as Queue


class EfficientUI(object):
    def __init__(self):
        # 相关参数
        img_fifo_path = "/tmp/dkimg.fifo"
        mask_fifo_path = "/tmp/dkmask.fifo"
        rgb_fifo_path = "/tmp/dkrgb.fifo"
        # 创建队列用于链接各个线程
        rgb_img_queue, spec_img_queue = Queue(), Queue()
        detector_queue, save_queue, self.visual_queue = Queue(), Queue(), Queue()
        mask_queue = Queue()
        # 两个接收者,接收光谱和rgb图像
        spec_len = Config.nRows * Config.nCols * Config.nBands * 4  # float型变量, 4个字节
        rgb_len = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
        spec_receiver = transmit.FifoReceiver(fifo_path=img_fifo_path, output=spec_img_queue, read_max_num=spec_len)
        rgb_receiver = transmit.FifoReceiver(fifo_path=rgb_fifo_path, output=rgb_img_queue, read_max_num=rgb_len)
        # 指令执行与图像流向控制
        subscribers = {'detector': detector_queue, 'visualize': self.visual_queue, 'save': save_queue}
        cmd_img_controller = transmit.CmdImgSplitMidware(rgb_queue=rgb_img_queue, spec_queue=spec_img_queue,
                                                         subscribers=subscribers)
        # 探测器
        detector = transmit.ThreadDetector(input_queue=detector_queue, output_queue=mask_queue)
        # 发送
        sender = transmit.FifoSender(output_fifo_path=mask_fifo_path, source=mask_queue)
        # 启动所有线程
        spec_receiver.start(post_process_func=transmit.BeforeAfterMethods.spec_data_post_process, name='spce_thread')
        rgb_receiver.start(post_process_func=transmit.BeforeAfterMethods.rgb_data_post_process, name='rgb_thread')
        cmd_img_controller.start(name='control_thread')
        detector.start(name='detector_thread')
        sender.start(pre_process=transmit.BeforeAfterMethods.mask_preprocess, name='sender_thread')

    def start(self):
        # 启动图形化
        while True:
            cv2.imshow('image_show', mat=np.ones((256, 1024)))
            key_code = cv2.waitKey(30)
            if key_code == ord("s"):
                pass


if __name__ == '__main__':
    app = EfficientUI()
    app.start()
