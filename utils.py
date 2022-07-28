# -*- codeing = utf-8 -*-
# Time : 2022/7/18 9:46
# @Auther : zhouchao
# @File: utils.py
# @Software:PyCharm
import glob
import os
from queue import Queue

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


class ImgQueue(Queue):
    """
        A custom queue subclass that provides a :meth:`clear` method.
    """

    def clear(self):
        """
        Clears all items from the queue.
        """

        with self.mutex:
            unfinished = self.unfinished_tasks - len(self.queue)
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
            self.queue.clear()
            self.not_full.notify_all()

    def safe_put(self, *args, **kwargs):
        if self.full():
            _ = self.get()
            return False
        self.put(*args, **kwargs)
        return True


def read_labeled_img(dataset_dir: str, color_dict: dict, ext='.bmp', is_ps_color_space=True) -> dict:
    """
    根据dataset_dir下的文件创建数据集

    :param dataset_dir: 文件夹名称，文件夹内必须包含'label'和'label'两个文件夹，并分别存放同名的图像与标签
    :param color_dict: 进行标签图像的颜色查找
    :param ext: 图片后缀名,默认为.bmp
    :param is_ps_color_space: 是否使用ps的标准lab色彩空间，默认True
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
        if is_ps_color_space:
            alpha, beta = np.array([100 / 255, 1, 1]), np.array([0, -128, -128])
            img = img * alpha + beta
            img = np.asarray(np.round(img, 0), dtype=int)
        dataset = {label: img[np.all(label_img == color, axis=2)] for color, label in color_dict.items()}
        total_dataset.merge(dataset)
    return total_dataset


def lab_scatter(dataset: dict, class_max_num=None, is_3d=False, is_ps_color_space=True, **kwargs):
    """
    在lab色彩空间内绘制3维数据分布情况

    :param dataset: 字典形式的数据集{label: vector(n x 3)},vector为lab色彩空间
    :param class_max_num: 每个类别最多画的样本数量，默认不限制
    :param is_3d: 进行lab三维绘制或者a,b两通道绘制
    :param is_ps_color_space: 是否使用ps的标准lab色彩空间，默认True
    :return: None
    """
    # 观察色彩分布情况
    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    for label, data in dataset.items():
        if class_max_num is not None:
            assert isinstance(class_max_num, int)
            if data.shape[0] > class_max_num:
                sample_idx = np.arange(data.shape[0])
                sample_idx = np.random.choice(sample_idx, class_max_num)
                data = data[sample_idx, :]
        l, a, b = [data[:, i] for i in range(3)]
        if is_3d:
            ax.scatter(a, b, l, label=label, alpha=0.1)
        else:
            ax.scatter(a, b, label=label, alpha=0.1)
    x_max, x_min, y_max, y_min, z_max, z_min = [127, -127, 127, -127, 100, 0] if is_ps_color_space else \
        [255, 0, 255, 0, 255, 0]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    if is_3d:
        ax.set_zlim(z_min, z_max)
        ax.set_zlabel('L')
    plt.legend()
    plt.show()


def size_threshold(img, blk_size, threshold):
    mask = img.reshape(img.shape[0], img.shape[1] // blk_size, blk_size).sum(axis=2). \
        reshape(img.shape[0] // blk_size, blk_size, img.shape[1] // blk_size).sum(axis=1)
    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1
    return mask


if __name__ == '__main__':
    color_dict = {(0, 0, 255): "yangeng", (255, 0, 0): "bejing", (0, 255, 0): "hongdianxian",
                  (255, 0, 255): "chengsebangbangtang", (0, 255, 255): "lvdianxian"}
    dataset = read_labeled_img("data/dataset", color_dict=color_dict, is_ps_color_space=False)
    lab_scatter(dataset, class_max_num=20000, is_3d=False, is_ps_color_space=False)
