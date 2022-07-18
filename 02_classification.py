#!/usr/bin/env python
# coding: utf-8

# # 模型的训练

# In[16]:


import numpy as np

from models import AnonymousColorDetector
from utils import read_labeled_img

# ## 读取数据与构建数据集

# In[17]:


data_dir = "data/dataset"
color_dict = {(0, 0, 255): "yangeng"}
dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)

# ## 模型训练

# In[18]:


# 定义一些常量
threshold = 5
node_num = 20
negative_sample_num = None  # None或者一个数字
world_boundary = np.array([0, 0, 0, 255, 255, 255])
# 对数据进行预处理
x = np.concatenate([v for k, v in dataset.items()], axis=0)
negative_sample_num = int(x.shape[0] * 0.7) if negative_sample_num is None else negative_sample_num

model = AnonymousColorDetector()

model.fit(x, world_boundary, threshold, negative_sample_size=negative_sample_num, train_size=0.7)

model.save()
