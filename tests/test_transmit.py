import logging
import time
import unittest

import numpy as np

import transmit
from config import Config
from transmit import FileReceiver, FifoReceiver, FifoSender
from utils import ImgQueue


class TransmitterTest(unittest.TestCase):
    def test_file_receiver(self):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info('测试文件接收器')
        image_queue = ImgQueue()
        file_receiver = FileReceiver(job_name='rgb img receive', input_dir='../data', output_queue=image_queue,
                                     speed=0.5, name_pattern=None)
        virtual_data = np.random.randint(0, 255, (1024, 4096, 3), dtype=np.uint8)
        file_receiver.start(need_time=True, virtual_data=virtual_data)
        for i in range(5):
            time_record, read_data, virtual_data_rec = image_queue.get()
            current_time = time.time()
            logging.info(f'Spent {(current_time - time_record) * 1000:.2f}ms to get image with shape {virtual_data.shape}')
            is_equal = np.all(virtual_data_rec == virtual_data, axis=(0, 1, 2))
            self.assertTrue(is_equal)
            self.assertEqual(virtual_data.shape, (1024, 4096, 3))
        file_receiver.stop()

    @unittest.skip('skip')
    def test_fifo_receiver_sender(self):
        total_rgb = Config.nRgbRows * Config.nRgbCols * Config.nRgbBands * 1  # int型变量
        image_queue, input_queue = ImgQueue(), ImgQueue()
        fifo_receiver = FifoReceiver(job_name='fifo img receive', fifo_path='/tmp/dkimg.fifo', output=image_queue,
                                     read_max_num=total_rgb)
        fifo_sender = FifoSender(fifo_path='/tmp/dkimg.fifo', source=input_queue, job_name='fifo img send')
        virtual_data = np.random.randint(0, 255, (1024, 4096, 3), dtype=np.uint8)
        fifo_sender.start(preprocess=transmit.BeforeAfterMethods.mask_preprocess)
        fifo_receiver.start()
        logging.debug('Start to send virtual data')
        for i in range(5):
            logging.debug('put data to input queue')

            input_queue.put(virtual_data)
            logging.debug('put data to input queue done')
            virtual_data = image_queue.get()

            # logging.info(f'Spent {(current_time - time_record) * 1000:.2f}ms to get image with shape {virtual_data.shape}')
            self.assertEqual(virtual_data.shape, (1024, 4096, 3))


if __name__ == '__main__':
    unittest.main()
