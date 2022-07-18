# -*- codeing = utf-8 -*-
# Time : 2022/7/18 9:46
# @Auther : zhouchao
# @File: utils.py
# @Software:PyCharm
import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class MergeDict(dict):
    def __init__(self):
        super(MergeDict, self).__init__()

    def merge(self, merged: dict):
        for k, v in merged.items():
            if k not in self.keys():
                self.update({k: v})
            else:
                original = self.__getitem__(k)
                new_value = np.concatenate([original, v], axis=0)
                self.update({k: new_value})
        return self


def read_labeled_img(dataset_dir: str, color_dict: dict, ext='.bmp') -> dict:
    """
    根据dataset_dir下的文件创建数据集

    :param dataset_dir: 文件夹名称，文件夹内必须包含'label'和'label'两个文件夹，并分别存放同名的图像与标签
    :param color_dict: 进行标签图像的颜色查找
    :param ext: 图片后缀名,默认为.bmp
    :return: 字典形式的数据集{label: vector(n x 3)},vector为lab色彩空间
    """
    img_names = [img_name for img_name in os.listdir(os.path.join(dataset_dir, 'label'))
                 if img_name.endswith(ext)]
    total_dataset = MergeDict()
    for img_name in img_names:
        img_path, label_path = [os.path.join(dataset_dir, folder, img_name) for folder in ['img', 'label']]
        # 读取图片和色彩空间转换
        img = cv2.imread(img_path)
        label_img = cv2.imread(label_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 从opencv的色彩空间到Photoshop的色彩空间
        alpha, beta = np.array([100 / 255, 1, 1]), np.array([0, -128, -128])
        img = img * alpha + beta
        img = np.asarray(np.round(img, 0), dtype=int)
        dataset = {label: img[np.all(label_img == color, axis=2)] for color, label in color_dict.items()}
        total_dataset.merge(dataset)
    return total_dataset


def lab_scatter(dataset: dict, class_max_num=None):
    """
    在lab色彩空间内绘制3维数据分布情况

    :param dataset: 字典形式的数据集{label: vector(n x 3)},vector为lab色彩空间
    :param class_max_num: 每个类别最多画的样本数量，默认不限制
    :return: None
    """
    # 观察色彩分布情况
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for label, data in dataset.items():
        if class_max_num is not None:
            assert isinstance(class_max_num, int)
            if data.shape[0] > class_max_num:
                sample_idx = np.arange(data.shape[0])
                sample_idx = np.random.choice(sample_idx, class_max_num)
                data = data[sample_idx, :]
        l, a, b = [data[:, i] for i in range(3)]
        ax.scatter(a, b, l, label=label, alpha=0.1)
    ax.set_xlim(-127, 127)
    ax.set_ylim(-127, 127)
    ax.set_zlim(0, 100)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = read_labeled_img("data/dataset", color_dict={(0, 0, 255): 1, (255, 0, 0): 2})
    lab_scatter(dataset, class_max_num=2000)
