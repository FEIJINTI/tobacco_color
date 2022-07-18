# -*- codeing = utf-8 -*-
# Time : 2022/7/18 14:03
# @Auther : zhouchao
# @File: models.py
# @Software:PyCharm、
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from elm import ELM


class AnonymousColorDetector(object):
    def __init__(self):
        self.model = None

    def fit(self, x: np.ndarray, world_boundary: np.ndarray, threshold: float, model_selected: str = 'elm',
            negative_sample_size: int = 1000, train_size: float = 0.8, **kwargs):
        """
        拟合到指定的样本分布情况下，根据x进行分布的变化。

        :param x: ndarray类型的正样本数据，给出的正样本形状为 n x feature_num
        :param world_boundary: 整个世界的边界，边界形状为 feature_num个下限, feature_num个上限
        :param threshold: 与正样本之间的距离阈值大于多少则不认为是指定的样本类别
        :param model_selected: 选择模型，默认为elm
        :param negative_sample_size: 负样本的数量
        :param kwargs: 与模型相对应的参数
        :return:
        """
        assert model_selected in ['elm']
        if model_selected == 'elm':
            node_num = kwargs.get('node_num', 10)
            self.model = ELM(input_size=x.shape[1], node_num=node_num, output_num=2, **kwargs)
        negative_samples = self.generate_negative_samples(x, world_boundary, threshold,
                                                          sample_size=negative_sample_size)
        data_x, data_y = np.concatenate([x, negative_samples], axis=0), \
                         np.concatenate([np.ones(x.shape[0], dtype=int),
                                         np.zeros(negative_samples.shape[0], dtype=int)], axis=0)
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, train_size=train_size, shuffle=True)

        self.model.fit(x_train, y_train)
        y_predict = self.model.predict(x_val)
        print(classification_report(y_true=y_val, y_pred=y_predict))

    def predict(self, x):
        return self.model.predict(x)

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
        while generated_sample_num <= sample_size:
            generated_data = np.random.uniform(world_boundary[:feature_num], world_boundary[feature_num:],
                                               size=(sample_size, feature_num))
            for sample_idx in range(generated_data.shape[0]):
                sample = generated_data[sample_idx, :]
                in_threshold = np.any(np.sum(np.power(sample - x, 2), axis=1) < threshold)
                if not in_threshold:
                    negative_samples[sample_idx, :] = sample
                    generated_sample_num += 1
                    if generated_sample_num >= sample_size:
                        break
        return negative_samples


if __name__ == '__main__':
    detector = AnonymousColorDetector()
    x = np.array([[10, 30, 20], [10, 35, 25], [10, 35, 36]])
    world_boundary = np.array([0, -127, -127, 100, 127, 127])
    detector.fit(x, world_boundary, threshold=5, negative_sample_size=2000)
