import os

import numpy as np

from config import Config
from models import Detector, AnonymousColorDetector, RgbDetector
import cv2

# 测试单张图片使用RGB进行预测的效果

# # 测试时间
# import time
# start_time = time.time()
# 读取图片
file_path = r"E:\Tobacco\data\testImgs\Image_2022_0726_1413_46_400-001165.bmp"
img = cv2.imread(file_path)[..., ::-1]
print("img.shape:", img.shape)

# 初始化和加载色彩模型
print('Initializing color model...')
rgb_detector = RgbDetector(tobacco_model_path=r'../weights/tobacco_dt_2022-08-27_14-43.model',
                           background_model_path=r"../weights/background_dt_2022-08-22_22-15.model",
                           ai_path='../weights/best0827.pt')
_ = rgb_detector.predict(np.ones((Config.nRgbRows, Config.nRgbCols, Config.nRgbBands), dtype=np.uint8) * 40)
print('Color model loaded.')

# 预测单张图片
print('Predicting...')
mask_rgb = rgb_detector.predict(img).astype(np.uint8)

# # 测试时间
# end_time = time.time()
# print("time cost:", end_time - start_time)

# 使用matplotlib展示两个图片的对比
import matplotlib.pyplot as plt
# 切换matplotlib的后端为qt，否则会报错
plt.switch_backend('qt5agg')

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].matshow(mask_rgb)
plt.show()




