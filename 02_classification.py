#!/usr/bin/env python
# coding: utf-8

# # 模型的训练

# In[16]:


import numpy as np
import scipy
from imblearn.under_sampling import RandomUnderSampler
from models import AnonymousColorDetector
from utils import read_labeled_img

# ## 读取数据与构建数据集

# In[17]:


data_dir = "data/dataset"
color_dict = {(0, 0, 255): "yangeng", (255, 0, 0): 'beijing'}
label_index = {"yangeng": 1, "beijing": 0}
dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)
rus = RandomUnderSampler(random_state=0)
x_list, y_list = np.concatenate([v for k, v in dataset.items()], axis=0).tolist(), \
                 np.concatenate([np.ones((v.shape[0],)) * label_index[k] for k, v in dataset.items()], axis=0).tolist()

x_resampled, y_resampled = rus.fit_resample(x_list, y_list)
dataset = {"inside": np.array(x_resampled)}

# ## 模型训练

# In[18]:


# 定义一些常量
threshold = 5
node_num = 20
negative_sample_num = None  # None或者一个数字
world_boundary = np.array([0, 0, 0, 255, 255, 255])
# 对数据进行预处理
x = np.concatenate([v for k, v in dataset.items()], axis=0)
negative_sample_num = int(x.shape[0] * 1.2) if negative_sample_num is None else negative_sample_num

model = AnonymousColorDetector()

model.fit(x, world_boundary, threshold, negative_sample_size=negative_sample_num, train_size=0.7,
          is_save_dataset=True, model_selection='dt')
# data = scipy.io.loadmat('dataset_2022-07-19_15-07.mat')
# x, y = data['x'], data['y'].ravel()
# model.fit(x, y=y, is_generate_negative=False, model_selection='dt')

model.save()
