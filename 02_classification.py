import numpy as np
import scipy
from imblearn.under_sampling import RandomUnderSampler
from models import AnonymousColorDetector
from utils import read_labeled_img

# %%
train_from_existed = False  # 是否从现有数据训练，如果是的话，那就从dataset_file训练，否则就用data_dir里头的数据
data_dir = "data/dataset"  # 数据集，文件夹下必须包含`img`和`label`两个文件夹，放置相同文件名的图片和label
dataset_file = "data/dataset/dataset_2022-07-20_10-04.mat"

color_dict = {(0, 0, 255): "yangeng", (255, 0, 0): 'beijing', (0, 255, 0): "zibian"}  # 颜色对应的类别
# color_dict = {(0, 0, 255): "yangeng"}
# color_dict = {(255, 0, 0): 'beijing'}
# color_dict = {(0, 255, 0): "zibian"}
label_index = {"yangeng": 1, "beijing": 0, "zibian": 2}  # 类别对应的序号
show_samples = True  # 是否展示样本

# 定义一些训练量
threshold = 5  # 正样本周围多大范围内的还算是正样本
node_num = 20  # 如果使用ELM作为分类器物，有多少的节点
negative_sample_num = None  # None或者一个数字，对应生成的负样本数量
# %% md
## 读取数据
# %%
dataset = read_labeled_img(data_dir, color_dict=color_dict, is_ps_color_space=False)
if show_samples:
    from utils import lab_scatter

    lab_scatter(dataset, class_max_num=30000, is_3d=True, is_ps_color_space=False)
# %% md
## 数据平衡化
# %%
if len(dataset) > 1:
    rus = RandomUnderSampler(random_state=0)
    x_list, y_list = np.concatenate([v for k, v in dataset.items()], axis=0).tolist(), \
                     np.concatenate([np.ones((v.shape[0],)) * label_index[k] for k, v in dataset.items()],
                                    axis=0).tolist()
    x_resampled, y_resampled = rus.fit_resample(x_list, y_list)
    dataset = {"inside": np.array(x_resampled)}
# %% md
## 模型训练
# %%
# 对数据进行预处理
x = np.concatenate([v for k, v in dataset.items()], axis=0)
negative_sample_num = int(x.shape[0] * 1.2) if negative_sample_num is None else negative_sample_num
model = AnonymousColorDetector()
# %%
if train_from_existed:
    data = scipy.io.loadmat(dataset_file)
    x, y = data['x'], data['y'].ravel()
    model.fit(x, y=y, is_generate_negative=False, model_selection='dt')
else:
    world_boundary = np.array([0, 0, 0, 255, 255, 255])
    model.fit(x, world_boundary, threshold, negative_sample_size=negative_sample_num, train_size=0.7,
              is_save_dataset=True, model_selection='dt')
model.save()