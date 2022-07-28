# -*- codeing = utf-8 -*-
# Time : 2022/7/18 14:03
# @Auther : zhouchao
# @File: models.py
# @Software:PyCharm、
import datetime
import pickle
from queue import Queue

import cv2
import numpy as np
import scipy.io
import threading
from scipy.ndimage import binary_dilation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from config import Config
from utils import lab_scatter, read_labeled_img, size_threshold


deploy = False
if not deploy:
    print("Training env")
    from tqdm import tqdm
    from elm import ELM


class Detector(object):
    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError


class AnonymousColorDetector(Detector):
    def __init__(self, file_path: str = None):
        super(AnonymousColorDetector, self).__init__()
        self.model = None
        self.model_type = 'None'
        if file_path is not None:
            self.load(file_path)

    def fit(self, x: np.ndarray, world_boundary: np.ndarray = None, threshold: float = None,
            is_generate_negative: bool = True, y: np.ndarray = None, model_selection='elm',
            negative_sample_size: int = 1000, train_size: float = 0.8, is_save_dataset=False, **kwargs):
        """
        拟合到指定的样本分布情况下，根据x进行分布的变化。

        :param x: ndarray类型的正样本数据，给出的正样本形状为 n x feature_num
        :param world_boundary: 整个世界的边界，边界形状为 feature_num个下限, feature_num个上限
        :param threshold: 与正样本之间的距离阈值大于多少则不认为是指定的样本类别
        :param is_generate_negative: 是否生成负样本
        :param y: 给出x对应的样本y
        :param model_selection: 模型的选择, in ['elm', 'decision tree']
        :param negative_sample_size: 负样本的数量
        :param train_size: 训练集的比例, float
        :param is_save_dataset: 是否保存数据集
        :param kwargs: 与模型相对应的参数
        :return:
        """
        if model_selection == 'elm':
            node_num = kwargs.get('node_num', 10)
            self.model = ELM(input_size=x.shape[1], node_num=node_num, output_num=2, **kwargs)
        elif model_selection == 'dt':
            self.model = DecisionTreeClassifier(**kwargs)
        else:
            raise ValueError("你看看我要的是啥")
        self.model_type = model_selection
        if is_generate_negative:
            negative_samples = self.generate_negative_samples(x, world_boundary, threshold,
                                                              sample_size=negative_sample_size)
            data_x, data_y = np.concatenate([x, negative_samples], axis=0), \
                             np.concatenate([np.ones(x.shape[0], dtype=int),
                                             np.zeros(negative_samples.shape[0], dtype=int)], axis=0)
        else:
            data_x, data_y = x, y
        if is_save_dataset:
            path = datetime.datetime.now().strftime("dataset_%Y-%m-%d_%H-%M.mat")
            scipy.io.savemat(path, {'x': data_x, 'y': data_y})
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, train_size=train_size, shuffle=True,
                                                          stratify=data_y)
        self.model.fit(x_train, y_train)
        y_predict = self.model.predict(x_val)
        print(classification_report(y_true=y_val, y_pred=y_predict))

    def predict(self, x, threshold_low=5, threshold_high=255):
        """
        输入rgb彩色图像

        :param x: rgb彩色图像,np.ndarray
        :return:
        """
        w, h = x.shape[1], x.shape[0]
        x = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        x = x.reshape(w * h, -1)
        mask = (threshold_low < x[:, 0]) & (x[:, 0] < threshold_high)
        result = np.ones((w * h,), dtype=np.uint8)

        if np.any(mask):
            mask_result = self.model.predict(x[mask])
            result[mask] = mask_result
        return result.reshape(h, w)

    @staticmethod
    def generate_negative_samples(x: np.ndarray, world_boundary: np.ndarray, threshold: float, sample_size: int):
        """
        根据正样本和世界边界生成负样本

        :param x: ndarray类型的正样本数据，给出的正样本形状为 n x feature_num
        :param world_boundary: 整个世界的边界，边界形状为 feature_num个下限, feature_num个上限, array like
        :param threshold: 与正样本x之间的距离限制
        :return: 负样本形状为：(sample_size, feature_num)
        """

        feature_num = x.shape[1]
        negative_samples = np.zeros((sample_size, feature_num), dtype=x.dtype)
        generated_sample_num = 0
        bar = tqdm(total=sample_size, ncols=100)
        while generated_sample_num <= sample_size:
            generated_data = np.random.uniform(world_boundary[:feature_num], world_boundary[feature_num:],
                                               size=(sample_size, feature_num))
            for sample_idx in range(generated_data.shape[0]):
                sample = generated_data[sample_idx, :]
                in_threshold = np.any(np.sum(np.power(sample - x, 2), axis=1) < threshold)
                if not in_threshold:
                    negative_samples[sample_idx, :] = sample
                    generated_sample_num += 1
                    bar.update()
                    if generated_sample_num >= sample_size:
                        break
        bar.close()
        return negative_samples

    def pretreatment(self, x):
        return cv2.resize(x, (1024, 256))

    def swell(self, x):
        return cv2.dilate(x, kernel=np.ones((3, 3), np.uint8))

    def save(self):
        path = datetime.datetime.now().strftime(f"{self.model_type}_%Y-%m-%d_%H-%M.model")
        with open(path, 'wb') as f:
            pickle.dump((self.model_type, self.model), f)

    def load(self, file_path):
        with open(file_path, 'rb') as model_file:
            data = pickle.load(model_file)
        self.model_type, self.model = data

    def visualize(self, world_boundary: np.ndarray, sample_size: int, ground_truth=None,
                  **kwargs):
        feature_num = world_boundary.shape[0] // 2
        x = np.random.uniform(world_boundary[:feature_num], world_boundary[feature_num:],
                              size=(sample_size, feature_num))
        pred_y = self.model.predict(x)
        draw_dataset = {'Inside': x[pred_y == 1, :], 'Outside': x[pred_y == 0, :]}
        if ground_truth is not None:
            draw_dataset.update({'Given': ground_truth})
        lab_scatter(draw_dataset, is_3d=True, is_ps_color_space=False, **kwargs)


class ManualTree:
    # 初始化机器学习像素模型、深度学习像素模型、分块模型
    def __init__(self, blk_model_path, pixel_model_path):
        self.pixel_model_ml = PixelModelML(pixel_model_path)
        self.blk_model = BlkModel(blk_model_path)

    # 区分烟梗和非黄色且非背景的杂质
    @staticmethod
    def is_yellow(features):
        features = features.reshape((Config.nRows * Config.nCols), len(Config.selected_bands))
        sum_x = features.sum(axis=1)[..., np.newaxis]
        rate = features / sum_x
        mask = ((rate < Config.is_yellow_max) & (rate > Config.is_yellow_min))
        mask = np.all(mask, axis=1).reshape(Config.nRows, Config.nCols)
        return mask

    # 区分背景和黄色杂质
    @staticmethod
    def is_black(feature, threshold):
        feature = feature.reshape((Config.nRows * Config.nCols), feature.shape[2])
        mask = (feature <= threshold)
        mask = np.all(mask, axis=1).reshape(Config.nRows, Config.nCols)
        return mask

    # 预测出烟梗的mask
    def predict_tobacco(self, x: np.ndarray) -> np.ndarray:
        """
        预测出烟梗的mask
        :param x: 图像数据，形状是 nRows x nCols x nBands
        :return: bool类型的mask，是否为烟梗, True为烟梗
        """
        black_res = self.is_black(x[..., Config.black_yellow_bands], Config.is_black_threshold)
        yellow_res = self.is_yellow(x[..., Config.black_yellow_bands])
        yellow_things = (~black_res) & yellow_res
        x_yellow = x[yellow_things, ...]
        tobacco = self.pixel_model_ml.predict(x_yellow[..., Config.green_bands])
        yellow_things[yellow_things] = tobacco
        return yellow_things

    # 预测出杂质的机器学习像素模型
    def pixel_predict_ml_dilation(self, data, iteration) -> np.ndarray:
        """
        预测出杂质的位置mask
        :param data: 图像数据，形状是 nRows x nCols x nBands
        :param iteration: 膨胀的次数
        :return: bool类型的mask，是否为杂质, True为杂质
        """
        black_res = self.is_black(data[..., Config.black_yellow_bands], Config.is_black_threshold)
        yellow_res = self.is_yellow(data[..., Config.black_yellow_bands])
        # non_yellow_things为异色杂质
        non_yellow_things = (~black_res) & (~yellow_res)
        # yellow_things为黄色物体(烟梗+杂质)
        yellow_things = (~black_res) & yellow_res
        # x_yellow为挑出的黄色物体
        x_yellow = data[yellow_things, ...]
        if x_yellow.shape[0] == 0:
            return non_yellow_things
        else:
            tobacco = self.pixel_model_ml.predict(x_yellow[..., Config.green_bands]) > 0.5

            non_yellow_things[yellow_things] = ~tobacco
            # 杂质mask中将背景赋值为0,将杂质赋值为1
            non_yellow_things = non_yellow_things + 0

            # 烟梗mask中将背景赋值为0,将烟梗赋值为2
            yellow_things[yellow_things] = tobacco
            yellow_things = yellow_things + 0
            yellow_things = binary_dilation(yellow_things, iterations=iteration)
            yellow_things = yellow_things + 0
            yellow_things[yellow_things == 1] = 2

            # 将杂质mask和烟梗mask相加,得到的mask中含有0(背景),1(杂质),2(烟梗),3(膨胀后的烟梗与杂质相加的部分)
            mask = non_yellow_things + yellow_things
            mask[mask == 0] = False
            mask[mask == 1] = True
            mask[mask == 2] = False
            mask[mask == 3] = False
            return mask

    # 预测出杂质的分块模型
    def blk_predict(self, data):
        blk_result_array = self.blk_model.predict(data)
        return blk_result_array


# 机器学习像素模型类
class PixelModelML:
    def __init__(self, pixel_model_path):
        with open(pixel_model_path, "rb") as f:
            self.dt = pickle.load(f)

    def predict(self, feature):
        pixel_result_array = self.dt.predict(feature)
        return pixel_result_array


# 分块模型类
class BlkModel:
    def __init__(self, blk_model_path):
        self.rfc = None
        self.load(blk_model_path)

    @staticmethod
    def split_x(data: np.ndarray, blk_sz: int) -> list:
        """
        Split the data into slices for classification.将数据划分为多个像素块,便于后续识别.

        ;param data: image data, shape (num_rows x ncols x num_channels)
        ;param blk_sz: block size
        ;param sensitivity: 最少有多少个杂物点能够被认为是杂物
        ;return data_x, data_y: sliced data x (block_num x num_charnnels x blk_sz x blk_sz)
        """
        x_list = []
        for i in range(0, 256 // blk_sz):
            for j in range(0, 1024 // blk_sz):
                block_data = data[i * blk_sz: (i + 1) * blk_sz, j * blk_sz: (j + 1) * blk_sz, ...]
                x_list.append(block_data)
        return x_list

    def predict(self, data):
        data_blk = data
        data_blk = np.array(self.split_x(data_blk, blk_sz=Config.blk_size))
        data_blk = data_blk.reshape((data_blk.shape[0]), -1)
        y_pred = self.rfc.predict(data_blk)
        y_pred[y_pred < 2] = 0
        y_pred[y_pred > 1] = 1
        blk_result_array = y_pred.reshape(256 // Config.blk_size, 1024 // Config.blk_size).repeat(Config.blk_size,
                                                                                                  axis=0).repeat(
            Config.blk_size,
            axis=1)
        return blk_result_array

    def load(self, model_path: str):
        with open(model_path, "rb") as f:
            self.rfc = pickle.load(f)


class RgbDetector(Detector):
    def __init__(self, tobacco_model_path, background_model_path):
        self.background_detector = None
        self.tobacco_detector = None
        self.load(tobacco_model_path, background_model_path)

    def predict(self, rgb_data):
        rgb_data = self.tobacco_detector.pretreatment(rgb_data)  # resize to the required size
        background = self.background_detector.predict(rgb_data)
        tobacco = self.tobacco_detector.predict(rgb_data)
        tobacco_d = self.tobacco_detector.swell(tobacco)  # dilate the tobacco to remove the tobacco edge error
        high_s = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2HSV)[..., 1] > Config.threshold_s
        non_tobacco_or_background = 1 - (background | tobacco_d)  # 既非烟梗也非背景的区域
        rgb_predict_result = high_s | non_tobacco_or_background  # 高饱和度区域或者是双非区域都是杂质
        mask_rgb = size_threshold(rgb_predict_result, Config.blk_size, Config.rgb_size_threshold)  # 杂质大小限制，超过大小的才打
        return mask_rgb

    def load(self, tobacco_model_path, background_model_path):
        self.tobacco_detector = AnonymousColorDetector(tobacco_model_path)
        self.background_detector = AnonymousColorDetector(background_model_path)

    def save(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass


class SpecDetector(Detector):
    # 初始化机器学习像素模型、深度学习像素模型、分块模型
    def __init__(self, blk_model_path, pixel_model_path):
        self.blk_model = None
        self.pixel_model_ml = None
        self.load(blk_model_path, pixel_model_path)

    def load(self, blk_model_path, pixel_model_path):
        self.pixel_model_ml = PixelModelML(pixel_model_path)
        self.blk_model = BlkModel(blk_model_path)

    def predict(self, img_data):
        pixel_predict_result = self.pixel_predict_ml_dilation(data=img_data, iteration=1)
        blk_predict_result = self.blk_predict(data=img_data)
        mask = (pixel_predict_result & blk_predict_result).astype(np.uint8)
        mask = size_threshold(mask, Config.blk_size, Config.spec_size_threshold)
        return mask

    def save(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    # 区分烟梗和非黄色且非背景的杂质
    @staticmethod
    def is_yellow(features):
        features = features.reshape((Config.nRows * Config.nCols), len(Config.selected_bands))
        sum_x = features.sum(axis=1)[..., np.newaxis]
        rate = features / sum_x
        mask = ((rate < Config.is_yellow_max) & (rate > Config.is_yellow_min))
        mask = np.all(mask, axis=1).reshape(Config.nRows, Config.nCols)
        return mask

    # 区分背景和黄色杂质
    @staticmethod
    def is_black(feature, threshold):
        feature = feature.reshape((Config.nRows * Config.nCols), feature.shape[2])
        mask = (feature <= threshold)
        mask = np.all(mask, axis=1).reshape(Config.nRows, Config.nCols)
        return mask

    # 预测出烟梗的mask
    def predict_tobacco(self, x: np.ndarray) -> np.ndarray:
        """
        预测出烟梗的mask
        :param x: 图像数据，形状是 nRows x nCols x nBands
        :return: bool类型的mask，是否为烟梗, True为烟梗
        """
        black_res = self.is_black(x[..., Config.black_yellow_bands], Config.is_black_threshold)
        yellow_res = self.is_yellow(x[..., Config.black_yellow_bands])
        yellow_things = (~black_res) & yellow_res
        x_yellow = x[yellow_things, ...]
        tobacco = self.pixel_model_ml.predict(x_yellow[..., Config.green_bands])
        yellow_things[yellow_things] = tobacco
        return yellow_things

    # 预测出杂质的机器学习像素模型
    def pixel_predict_ml_dilation(self, data, iteration) -> np.ndarray:
        """
        预测出杂质的位置mask
        :param data: 图像数据，形状是 nRows x nCols x nBands
        :param iteration: 膨胀的次数
        :return: bool类型的mask，是否为杂质, True为杂质
        """
        black_res = self.is_black(data[..., Config.black_yellow_bands], Config.is_black_threshold)
        yellow_res = self.is_yellow(data[..., Config.black_yellow_bands])
        # non_yellow_things为异色杂质
        non_yellow_things = (~black_res) & (~yellow_res)
        # yellow_things为黄色物体(烟梗+杂质)
        yellow_things = (~black_res) & yellow_res
        # x_yellow为挑出的黄色物体
        x_yellow = data[yellow_things, ...]
        if x_yellow.shape[0] == 0:
            return non_yellow_things
        else:
            tobacco = self.pixel_model_ml.predict(x_yellow[..., Config.green_bands]) > 0.5

            non_yellow_things[yellow_things] = ~tobacco
            # 杂质mask中将背景赋值为0,将杂质赋值为1
            non_yellow_things = non_yellow_things + 0

            # 烟梗mask中将背景赋值为0,将烟梗赋值为2
            yellow_things[yellow_things] = tobacco
            yellow_things = yellow_things + 0
            yellow_things = binary_dilation(yellow_things, iterations=iteration)
            yellow_things = yellow_things + 0
            yellow_things[yellow_things == 1] = 2

            # 将杂质mask和烟梗mask相加,得到的mask中含有0(背景),1(杂质),2(烟梗),3(膨胀后的烟梗与杂质相加的部分)
            mask = non_yellow_things + yellow_things
            mask[mask == 0] = False
            mask[mask == 1] = True
            mask[mask == 2] = False
            mask[mask == 3] = False
            return mask

    # 预测出杂质的分块模型
    def blk_predict(self, data):
        blk_result_array = self.blk_model.predict(data)
        return blk_result_array


if __name__ == '__main__':
    data_dir = "data/dataset"
    color_dict = {(0, 0, 255): "yangeng"}
    dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)
    ground_truth = dataset['yangeng']
    detector = AnonymousColorDetector(file_path='models/dt_2022-07-19_14-38.model')
    # x = np.array([[10, 30, 20], [10, 35, 25], [10, 35, 36]])
    boundary = np.array([0, 0, 0, 255, 255, 255])
    # detector.fit(x, world_boundary, threshold=5, negative_sample_size=2000)
    detector.visualize(boundary, sample_size=50000, class_max_num=5000, ground_truth=ground_truth)
    temp = scipy.io.loadmat('data/dataset_2022-07-19_11-35.mat')
    x, y = temp['x'], temp['y']
    dataset = {'inside': x[y.ravel() == 1, :], "outside": x[y.ravel() == 0, :]}
    lab_scatter(dataset, class_max_num=5000, is_3d=True, is_ps_color_space=False)
