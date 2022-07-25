# 烟梗彩色相机识别

2022年7月18日开始开发的项目，使用彩色相机进行烟梗颜色的识别。

## 如何进行模型训练和部署？
1. 项目当中需要包含`data`和`models`这两个文件夹，请下载到当前文件夹下,这是链接：[data](https://macrosolid-my.sharepoint.com/personal/feijinti_miaow_fun/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ffeijinti%5Fmiaow%5Ffun%2FDocuments%2FPycharmProjects%2Ftobacco%5Fcolor%2Fdata&ga=1), [models](https://macrosolid-my.sharepoint.com/:f:/g/personal/feijinti_miaow_fun/EiyBjWEX90JGn8S-e5Kh7N8B1GWvfvDcNbpleWDTwkDm1w?e=wyL4EF)
2. 使用[01_dataset.ipynb](./01_dataset.ipynb) 进行数据集的分析文件格式需要设置为这种形式：
    ```text
    dataset
    ├── label
    │   ├── img1.bmp
    │   └── ...
    └── img
        ├── img1.bmp
        └── ...
    ```
3. 使用[02_classification.ipynb](./02_classification.ipynb)进行训练
4. 使用[03_data_update.ipynb](02_classification.ipynb)进行数据的更新与添加
5. 使用`main_test.py`文件进行读图测试
6. **部署**，复制`utils.py`、`models.py`、`main.py`、`models/`、`config.py`这5个文件或文件夹，运行main.py来提供预测服务。


## 训练的原理
为了应对工业环境当中负样本少的特点，我们结合颜色有限空间的特性对我们的训练过程进行了优化，核心的优化方式在于制造负样本。

### 传统的负样本是怎么来的？

传统方法中，我们通过收集负样本，得到这样的结果：

![image-20220721170840283](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721170840283.png)

这张图里，负样本是紫色、红色和绿色区域，正样本则是蓝色区域。如果用这样的数据进行训练，我们得到的模型一定是欠约束的，因为我们不知道没有负样本的区域上，模型到底会表现出怎样的结果，所有我们想到，去手动制造负样本。

### 负样本是怎么造出来的？

 我们对于一个给定的色彩空间进行随机的生成一些数据，然后判断它是否是给定的正样本附近，如果是在附近，那么我们就把这些点看作是正样本，如果离得比较远，那么就会被当作是负样本。

### 训练的结果

添加负样本后进行训练，模型就会被约束在我们给定的样本范围内，就像下图看到的这样。

![image-20220721153751824](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721153751824.png)

在这里，绿色就是目标（烟梗）的色彩范围，橙色的和蓝色则表明了模型的判定范围，模型认为蓝色的区域就是烟梗，而橙色的区域就不是烟梗。

可以看到，蓝色区域与绿色区域是高度重叠的，并且蓝色比绿色区域要大一些的，这正是我们想要的效果。这表明模型对于烟梗的颜色有适度的宽容，允许色彩有一定的偏差，但大体上是要达到烟梗颜色范围内的。

这样的好处在于，即使出现了新的杂质，只要这些杂质的色彩不在模型的宽容范围内（蓝色范围内），那么都会被判定为杂质。

我们甚至可以直接将需要的目标色彩(烟梗的颜色)和背景的颜色直接合并到一类，这样还可以节省后续的各种逻辑判断、与、或、非等操作。

预测出来的结果也相对理想，就像下边两张图展示的这样。

![image-20220721201451562](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721201451562.png)

上图里可以看到烟梗和背景的错误识别点还是存在的，只是较为零散。

![image-20220721201617599](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721201617599.png)

而从这张图就可以看出各类杂质识别还是没问题的。

接下来我们更换采集所使用的镜头，得到更宽广的视角范围并观察其变化。

## 镜头的影响

我们现有的镜头包括广角和窄角两个，这两个镜头有着不同的成像效果，如下图所示。

| 视角     |                           广角镜头                           |                           窄角镜头                           |
| -------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 普通视角 | ![image-20220721155919349](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721155919349.png) | ![image-20220721155848160](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721155848160.png) |
| 放大视角 | ![image-20220721161109746](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721161109746.png) | ![image-20220721161117435](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721161117435.png) |

可以看到，镜头的影响有两方面

- 镜头会影响到物体的亮度，广角镜头下的物体看起来更暗
- 镜头会影响到物体的色彩，广角镜头下物体的色彩相比于窄角镜头的色彩饱和度要差很多，色相也有一定的偏移。

综合这两点来看，在条件允许的情况下应该选择窄角镜头。

但是我们这里的传送带非常宽，我们也只有1个相机，不像陶朗一样一条线有29个相机，或者合肥的一条线4～8个相机，所以我们只能冒险试试了。

尝试的结果很糟糕，可以看到如下图所示：

![image-20220721164306227](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721164306227.png)

几乎所有的烟梗都被当成了杂质。

为了探究这个问题，我们回到色彩空间中进行观察，可以看到这样的结果：

![image-20220721164406435](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721164406435.png)

在新的镜头下，数据的分布和原本的数据已经发生了巨大的变化，根本就不是同一种东西，这就是为什么分类结果会出错了。

## 预测过程的后处理（异色问题）

### 问题的发现

![image-20220721154731187](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721154731187.png)

在摄影过程中，由于相机、镜头和拍摄物体多方面的原因会出现色散边的现象，就像上图这样本来应该黄色的烟梗，边缘却变成了暗绿色或紫色的。

这是由于不同波长的光折射率不同，到达成像单元的位置会出现细小的偏差，而我们的成像单元又比较的细小，举例来说，这可能使得物体上同样的点发出的红光到了1号像素，而发出的绿光本来应该也射到1号像素却射到了相邻的2号像素，这就导致色彩不对了。

根据资料，一般的解决方案是对于不同波长的光进行折射率补偿，使用抗色散镜头。但是由于条件有限，我们这里就只能用算法的形式硬抗这些误差了。

### 解决方案

- 一方面，用像素块的结果进行求和，像素块求和结果大于阈值的时候判定为杂质，这个方案目前正在使用，可以有效去除小块的错误识别点，也不会因为腐蚀操作导致缺损。
- 另一方面，进行烟梗结果的膨胀，去掉紫边。

## 模型的更新

### 如何应对新的目标物？

我们可以认为镜头的更换会直接让色彩发生特别离谱的变化，如果出现这样的情况，我们应该直接抛弃旧的数据，训练新的模型。这样也能获得相对不错的分类结果。

**所以我们今天晚上要丢弃旧的数据，进行这个新的模型的训练**。

在对这个模型进行重新训练后，获得的新模型识别结果如下图所示：

|     类别     |                             图像                             |
| :----------: | :----------------------------------------------------------: |
|     原图     | ![QQ20220721-1](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/QQ20220721-1.png) |
|   识别结果   | ![image-20220721203004305](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220721203004305.png) |
| 模型约束情况 | ![image-20220722095620511](https://raw.githubusercontent.com/Karllzy/imagebed/main/img/image-20220722095620511.png) |

