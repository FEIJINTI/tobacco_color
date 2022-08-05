import torch
import numpy as np


class Config:
    # 文件相关参数
    nRows, nCols, nBands = 256, 1024, 22
    nRgbRows, nRgbCols, nRgbBands = 1024, 4096, 3

    # 需要设置的谱段等参数
    selected_bands = [127, 201, 202, 294]
    bands = [127, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
             211, 212, 213, 214, 215, 216, 217, 218, 219, 294]
    is_yellow_min = np.array([0.10167048, 0.1644719, 0.1598884, 0.31534621])
    is_yellow_max = np.array([0.212984, 0.25896924, 0.26509268, 0.51943593])
    is_black_threshold = np.asarray([0.1369, 0.1472, 0.1439, 0.1814])
    black_yellow_bands = [0, 2, 3, 21]
    green_bands = [i for i in range(1, 21)]

    # 光谱模型参数
    blk_size = 4  # 必须是2的倍数，不然会出错
    pixel_model_path = r"./models/pixel_2022-08-02_15-22.model"
    blk_model_path = r"./models/rf_4x4_c22_20_sen8_9.model"
    spec_size_threshold = 3

    # rgb模型参数
    rgb_tobacco_model_path = r"models/tobacco_dt_2022-07-26_15-57.model"
    rgb_background_model_path = r"models/background_dt_2022-07-27_08-11.model"
    threshold_low, threshold_high = 10, 230
    threshold_s = 175
    rgb_size_threshold = 4

    # mask parameter
    target_size = (1024, 1024)  # (Width, Height) of mask

    # save part
    offset_vertical = 0

