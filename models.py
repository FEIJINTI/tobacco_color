# -*- codeing = utf-8 -*-
# Time : 2022/7/18 14:03
# @Auther : zhouchao
# @File: models.py
# @Software:PyCharm、
import datetime
import pickle

import cv2
import numpy as np
import scipy.io
import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import lab_scatter, read_labeled_img
from tqdm import tqdm

from elm import ELM


class Detector(object):
    def __int__(self, *args, **kwargs):
        raise NotImplementedError

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

    def predict(self, x):
        """
        输入rgb彩色图像

        :param x: rgb彩色图像,np.ndarray
        :return:
        """
        w, h = x.shape[1], x.shape[0]
        x = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
        result = self.model.predict(x.reshape(w * h, -1))
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


if __name__ == '__main__':
    data_dir = "data/dataset"
    color_dict = {(0, 0, 255): "yangeng"}
    dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)
    ground_truth = dataset['yangeng']
    detector = AnonymousColorDetector(file_path='models/dt_2022-07-19_14-38.model')
    # x = np.array([[10, 30, 20], [10, 35, 25], [10, 35, 36]])
    world_boundary = np.array([0, 0, 0, 255, 255, 255])
    # detector.fit(x, world_boundary, threshold=5, negative_sample_size=2000)
    detector.visualize(world_boundary, sample_size=50000, class_max_num=5000, ground_truth=ground_truth)
    data = scipy.io.loadmat('data/dataset_2022-07-19_11-35.mat')
    x, y = data['x'], data['y']
    dataset = {'inside': x[y.ravel() == 1, :], "outside": x[y.ravel() == 0, :]}
    lab_scatter(dataset, class_max_num=5000, is_3d=True, is_ps_color_space=False)
