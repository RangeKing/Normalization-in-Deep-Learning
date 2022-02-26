# 归一化方法大盘点

## 0 导言

本文第1节将会介绍`归一化基础知识点`，2节介绍4种深度学习中归一化方法——[BN](#2-Batch-Normalization（BN）)、[LN](#3-Layer-Normalization（LN）)、[IN](#4-Instance-Normalization（IN）)和[GN](#5-Group-Normalization（GN）)的概念及对应的PaddleAPI，在第3节总结不同方法的的适用范围。

未涉及其他归一化方法可以参见第4节 `扩展阅读`。

## 1 归一化基础知识点<sup>[1]</sup>

### 1.1 什么是归一化？

归一化是一种数据处理方式，能将数据经过处理后限制在某个固定范围内。

归一化存在两种形式，一种是在通常情况下，将数处理为 [0, 1] 之间的小数，其目的是为了在随后的数据处理过程中更便捷。例如，在图像处理中，就会将图像从 [0, 255] 归一化到 [0, 1]之间，这样既不会改变图像本身的信息储存，又可加速后续的网络处理。其他情况下，也可将数据处理到 [-1, 1] 之间，或其他的固定范围内。另一种是通过归一化将有量纲表达式变成无量纲表达式。那么什么是量纲，又为什么需要将有量纲转化为无量纲呢？具体举一个例子。当我们在做对房价的预测时，收集到的数据中，如房屋的面积、房间的数量、到地铁站的距离、住宅附近的空气质量等，都是量纲，而他们对应的量纲单位分别为平方米、个数、米、AQI等。这些量纲单位的不同，导致数据之间不具有可比性。同时，对于不同的量纲，数据的数量级大小也是不同的，比如房屋到地铁站的距离可以是上千米，而房屋的房间数量一般只有几个。经过归一化处理后，不仅可以消除量纲的影响，也可将各数据归一化至同一量级，从而解决数据间的可比性问题。

> **注**：本文中第1节`归一化基础知识点`中的所提及的`归一化`是一个比较泛化的概念，其余章节提及的`归一化`基本上都特指`Z-score 归一化`。

### 1.2 为什么要归一化？

1. 如在`1.1 什么是归一化`中所讲，归一化可以将有量纲转化为无量纲，同时将数据归一化至同一量级，解决数据间的可比性问题。在回归模型中，自变量的量纲不一致会导致回归系数无法解读或错误解读。在KNN、Kmeans等需要进行距离计算的算法中，量纲的量级不同可能会导致拥有较大量级的特征在进行距离计算时占主导地位，从而影响学习结果。

2. 数据归一化后，寻求最优解的过程会变得平缓，可以更快速的收敛到最优解。详解请参见 `1.3 为什么归一化能提高求解最优解的速度`。

### 1.3 为什么归一化能提高求解最优解的速度?

在`1.1 什么是归一化`中，我们提到一个对房价进行预测的例子，假设自变量只有房子到地铁站的距离<img src="https://latex.codecogs.com/svg.latex?x_{1}">和房子内房间的个数<img src="https://latex.codecogs.com/svg.latex?x_{2}">，因变量为房价，预测公式和损失函数分别为：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?y=\theta_{1}x_{1}+\theta_{2}x_{2}\\J=(\theta_{1}x_{1}+\theta_{2}x_{2}-y_{label})^2"></div>


在未归一化时，房子到地铁站的距离的取值在0～5000之间，而房间个数的取值范围仅为0～10。假设<img src="https://latex.codecogs.com/svg.latex?x_{1} = 1000,x_{2} = 3">， 那么损失函数的公式可以写为：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?J=(1000\theta_{1}+3\theta_{2}-y_{label})^2"></div>

可将该损失函数寻求最优解过程可视化为下图：
![](https://paddlepedia.readthedocs.io/en/latest/_images/normalization.png)

<div align=center>图1: 损失函数的等高线，（左）为未归一化时，（右）为归一化</div><br>

在图1中，左图的红色椭圆代表归一化前的损失函数等高线，蓝色线段代表梯度的更新，箭头的方向代表梯度更新的方向。寻求最优解的过程就是梯度更新的过程，其更新方向与登高线垂直。由于<img src="https://latex.codecogs.com/svg.latex?x_1"> 和 <img src="https://latex.codecogs.com/svg.latex?x_2"> 的量级相差过大，损失函数的等高线呈现为一个瘦窄的椭圆。因此如图1（左）所示，瘦窄的椭圆形会使得梯度下降过程呈之字形呈现，导致梯度下降速度缓慢。

当数据经过归一化后，<img src="https://latex.codecogs.com/svg.latex?x_{1}^{'}=\frac{1000-0}{5000-0}=0.2">，<img src="https://latex.codecogs.com/svg.latex?x_{2}^{'}=\frac{3-0}{10-0}=0.3">，那么损失函数的公式可以写为：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?J(x)=(0.2\theta_{1}+0.3\theta_{2}-y_{label})^2"></div>

我们可以看到，经过归一化后的数据属于同一量级，损失函数的等高线呈现为一个矮胖的椭圆形（如图1（右）所示），求解最优解过程变得更加迅速且平缓，因此可以在通过梯度下降进行求解时获得更快的收敛。

## 1.4 归一化有哪些类型

1. Min-max Normalization (Rescaling)：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{x-min(x)}{max(x)-min(x)}"></div>

   归一化后的数据范围为 [0, 1]，其中 <img src="https://latex.codecogs.com/svg.latex?min(x)">、 分别求样本数据的最小值和最大值。

2. Mean Normalization：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{x-mean(x)}{max(x)-min(x)}"></div>

   归一化后的数据范围为 [-1, 1]，其中 <img src="https://latex.codecogs.com/svg.latex?mean(x)"> 为样本数据的平均值。

   

3. Z-score Normalization (Standardization)：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{x-\mu}{\sigma}"></div>

   归一化后的数据均值为0、标准差为1，其中 <img src="https://latex.codecogs.com/svg.latex?\mu">、<img src="https://latex.codecogs.com/svg.latex?\sigma"> 分别为样本数据的均值和标准差。

   > 深度神经网络或者深度学习领域中提到的归一化基本上指的是Z-score Normalization，也就是统计学中的`标准化`<sup>[2]</sup>。

4. 非线性归一化：

   * 对数归一化：

	<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{\lg{x}}{\lg{max(x)}}"></div>

   * 反正切函数归一化：

	<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\arctan(x)*\frac{2}{\pi}"></div>

   归一化后的数据范围为 [-1, 1]

   * 小数定标标准化（Demical Point Normalization）:

	<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{x}{10^j}"></div>

   归一化后的数据范围为 [-1, 1]，<img src="https://latex.codecogs.com/svg.latex?j"> 为使<img src="https://latex.codecogs.com/svg.latex?max(|x^{'}|)\textless1"> 的最小整数。

   


## 1.5 不同归一化的使用条件

1. Min-max归一化和mean归一化适合在最大最小值明确不变的情况下使用，比如图像处理时，灰度值限定在 [0, 255] 的范围内，就可以用min-max归一化将其处理到[0, 1]之间。在最大最小值不明确时，每当有新数据加入，都可能会改变最大或最小值，导致归一化结果不稳定，后续使用效果也不稳定。同时，数据需要相对稳定，如果有过大或过小的异常值存在，min-max归一化和mean归一化的效果也不会很好。如果对处理后的数据范围有严格要求，也应使用min-max归一化或mean归一化。

2. Z-score归一化也可称为标准化，经过处理的数据呈均值为0，标准差为1的分布。在数据存在异常值、最大最小值不固定的情况下，可以使用标准化。标准化会改变数据的状态分布，但不会改变分布的种类。特别地，神经网络中经常会使用到z-score归一化，针对这一点，我们将在后续的文章中进行详细的介绍。

3. 非线性归一化通常被用在数据分化程度较大的场景，有时需要通过一些数学函数对原始值进行映射，如对数、反正切等。   


### 1.6 归一化（Normalization）和标准化（Standardization）的联系与区别

谈到归一化和标准化可能会存在一些概念的混淆，我们都知道归一化是指normalization，标准化是指standardization，但根据wiki上对feature scaling方法的定义，standardization其实就是z-score normalization，也就是说标准化其实是归一化的一种，而一般情况下，我们会把z-score归一化称为标准化，把min-max归一化简称为归一化。在下文中，我们也是用标准化指代z-score归一化，并使用归一化指代min-max归一化。

其实，归一化和标准化在本质上都是一种线性变换。在`1.4 归一化有哪些类型`中，我们提到了归一化和标准化的公式，对于归一化的公式，在数据给定的情况下，可以令<img src="https://latex.codecogs.com/svg.latex?a=max(x)-min(x),b=min(x)">，则归一化的公式可变形为：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?x^{'}=\frac{x-b}{a}=\frac{x}{a}-\frac{b}{a}=\frac{x}{a}-c"></div>

标准化的公式与变形后的归一化类似，其中的<img src="https://latex.codecogs.com/svg.latex?\mu">和<img src="https://latex.codecogs.com/svg.latex?\sigma">在数据给定的情况下，可以看作常数。因此，标准化的变形与归一化的类似，都可看作对<img src="https://latex.codecogs.com/svg.latex?x">按比例<img src="https://latex.codecogs.com/svg.latex?a">进行缩放，再进行<img src="https://latex.codecogs.com/svg.latex?c">个单位的平移。由此可见，归一化和标准化的本质都是一种线性变换，他们都不会因为对数据的处理而改变数据的原始数值排序。

那么归一化和标准化又有什么区别呢？

1. 归一化不会改变数据的状态分布，但标准化会改变数据的状态分布；
2. 归一化会将数据限定在一个具体的范围内，如 [0, 1]，但标准化不会，标准化只会将数据处理为均值为0，标准差为1。

> **归一化**（Normalization）和**标准化**（Standardization）是两种不同的数据规范方法。从传统统计学上严格地来说，深度学习中提到的**归一化**实际上是**标准化**。因为Batch Normalization论文发表的时候用的词是**归一化**，并且在深度学习领域应用广泛，所以后人也将类似的方法命名成**归一化**。

## 2 深度学习中的归一化

### 2.1 概述

神经网络中有各种归一化算法：Batch Normalization (BN)、Layer Normalization (LN)、Instance Normalization (IN)、Group Normalization (GN)。

从公式看它们都差不多：无非是减去均值，除以标准差，再施以线性映射。

<div align="center"><img src="https://latex.codecogs.com/svg.latex?y=\gamma\left(\frac{x-\mu(x)}{\sigma(x)}\right)+\beta"></div>

这些归一化算法的主要**区别在于操作的 feature map 维度不同**。如何区分并记住它们，一直是件令人头疼的事。参考知乎文章<sup>[3]</sup>，结合Paddle代码，介绍它们的具体操作，并给出一个方便记忆的类比。

### 2.2 批归一化Batch Normalization（BN）<sup>[4]</sup>

#### 2.2.1 提出背景
训练深层神经网络很复杂，因为在训练过程中，每层输入的分布会随着前几层的参数变化而变化。这就要求降低学习率和仔细的参数初始化，从而减慢了训练速度，使得训练具有饱和非线性的模型变得十分困难。

#### 2.2.2 概念及公式

Batch Normalization (BN) 是最早出现的，也通常是效果最好的归一化方式。特征图feature map：<img src="https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{N\times{C}\times{H}\times{W}}"> ,包含 N 个样本，每个样本通道数为 C，高为 H，宽为 W。对其求均值和方差时，将在 N、H、W上操作，而保留通道 C 的维度。具体来说，就是把第1个样本的第1个通道，加上第2个样本第1个通道 ...... 加上第 N 个样本第1个通道，求平均，得到通道 1 的均值（注意是除以 N×H×W 而不是单纯除以 N，最后得到的是一个代表这个 batch 第1个通道平均值的数字，而不是一个 H×W 的矩阵)。求通道 1 的方差也是同理。对所有通道都施加一遍这个操作，就得到了所有通道的均值和方差。

<div align="center"><img src="https://latex.codecogs.com/svg.latex?u_{c}(x)=\frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}\\\sigma_{c}(x)=\sqrt{\frac{1}{CHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W}\left(x_{nchw}-\mu_{c}(x)\right)^{2}+\epsilon}"></div>


![image-20220220102250130](https://raw.githubusercontent.com/RangeKing/Cloud-Image/main/img/202202201022870.png)

<div align=center>图2: CNN中BN示意图<br>注：图中蓝色表示一次BN处理的对象，···表示省略。之后的图3-5同理。</div><br>

如果把 <img src="https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{N\times{C}\times{H}\times{W}"> 类比为一叠相册。假设当N=3、C=4、H=5、B=6时，这叠相册总共有 N 本，每本有 C 页，每页照片的尺寸[长×宽]为[H×W]。BN 求均值时，相当于把这些相册按页码一一对应地加起来，再除以这些照片的像素总数：N×H×W。BN 求均值是一个“跨相册求平均”的操作（如图 2 所示），求标准差时也是同理。


#### 2.2.3 原理解析

BN论文原文给出的解释是BN可以解决神经网络训练过程中的ICS（Internal Covariate Shift）问题，所谓ICS问题，指的是由于深度网络由很多隐层构成，在训练过程中由于底层网络参数不断变化，导致上层隐层神经元激活值的分布逐渐发生很大的变化和偏移，而这非常不利于有效稳定地训练神经网络。

2018年的一篇文章[<sup>[5]</sup>](#refer-anchor-2)针对这个问题做了实验。他们得出的结论是：

1. ICS问题在较深的网络中确实是普遍存在的，但是这并非导致深层网络难以训练的根本原因。
2. BN没有解决了ICS问题。即使是应用了BN，网络隐层中的输出仍然存在严重的ICS问题。
3. 在BN层输出后人工加入噪音模拟ICS现象，并不妨碍BN的优秀表现。这两方面的证据互相佐证来看的话，其实侧面说明了BN和ICS问题并没什么关系。
4. BN起作用的真正原因是使得优化问题的解空间更平滑了。这确保梯度更具预测性，从而允许使用更大范围的学习率，实现更快的网络收敛。（目前的主流观点）

#### 2.2.4 BN代码实践
```python
class paddle.nn.BatchNorm(num_channels, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, dtype='float32', data_layout='NCHW', in_place=False, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, use_global_stats=False, trainable_statistics=False)
```

该接口用于构建 `BatchNorm` 类的一个可调用对象，具体参数详情参考[BatchNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#paddle.nn.BatchNorm)。

- 重要参数含义：

   - `num_channels` (int) - 指明输入 Tensor 的通道数量。

   - `act` (str, 可选) - 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 [激活函数](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/layers/activations.html#api-guide-activations) ，默认值为None。

- 输入输出形状：

    - input: 2-D或以上 的Tensor。

    - output: 和输入形状一样。

- 代码示例如下：

    ```python
    import paddle
    from paddle import nn
    import numpy as np
    
    # Fake input
    np.random.seed(123)
    x_data = np.random.random(size=(2, 2, 3, 2)).astype('float32')
    x = paddle.to_tensor(x_data)
    print('---------------[Input]---------------\n',x)
    # Batch Norm
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#batchnorm
    batch_norm = paddle.nn.BatchNorm(2)
    batch_norm_out = batch_norm(x)
    print('---------------[Batch Norm Output]---------------\n',batch_norm_out)
    ```

### 2.3 层归一化Layer Normalization（LN）<sup>[6]</sup>

#### 2.3.1 提出背景

一般的批归一化（Batch Normalization，BN）算法对mini-batch数据集过分依赖，无法应用到在线学习任务中（此时mini-batch数据集包含的样例个数为1），在递归神经网络（Recurrent neural network，RNN）中BN的效果也不明显 ，RNN多用于自然语言处理任务，网络在不同训练周期内输入的句子，句子长度往往不同，在RNN中应用BN时，在不同时间周期使用mini-batch数据集的大小都需要不同，计算复杂，而且如果一个测试句子比训练集中的任何一个句子都长，在测试阶段RNN神经网络预测性能会出现严重偏差。如果更改为使用层归一化，就可以有效的避免这个问题。

Batch Normalization 的一个缺点是**需要较大的 batchsize 才能合理估训练数据的均值和方差(横向计算)，这导致内存很可能不够用**，同时它也**很难应用在训练数据长度不同的 RNN 模型上**。

#### 2.3.2 概念及公式
Layer Normalization (LN) 的一个优势是不需要批训练，在单条数据内部就能归一化。
对于 <img src="https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{N\times{C}\times{H}\times{W}}"> , LN 对每个样本的 C、H、W 维度上的数据求均值和标准差，保留 N 维度。

<div align="center"><img src="https://latex.codecogs.com/svg.latex?u_{n}(x)=\frac{1}{CHW}\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}\\\sigma_{n}(x)=\sqrt{\frac{1}{CHW}\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W}\left(x_{nchw}-\mu_{n}(x)\right)^{2}+\epsilon}"></div>

![image-20220220102401871](https://raw.githubusercontent.com/RangeKing/Cloud-Image/main/img/202202201024372.png)

<div align=center>图3: CNN中LN示意图</div><br></br>

继续采用之前的类比，把一个 batch 的 feature 类比为一叠相册。LN 求均值时，相当于把每一本相册的所有像素加起来，再除以这本相册的像素总数：C×H×W，即求单本相册（图3蓝色所示）的平均像素值，求标准差时也是同理。

#### 2.3.3 LN代码实践

```python
class paddle.nn.LayerNorm(normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None)
```

该接口用于构建 `LayerNorm` 类的一个可调用对象，具体参数详情参考[LayerNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#cn-api-nn-layernorm)。

- 重要参数含义：

    - `normalized_shape`(int或list或tuple)  – 需规范化的shape，期望的输入shape为 [*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]] 。如果是单个整数，则此模块将在最后一个维度上规范化（此时最后一维的维度需与该参数相同）。
    
- 输入输出形状：

    - input: 2-D, 3-D, 4-D或5-D 的Tensor。

    - output: 和输入形状一样。

- 代码示例如下：

    ```python
    # Layer Norm
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#layernorm
    layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
    layer_norm_out = layer_norm(x)
    
    print('---------------[Layer Norm Output]---------------\n',layer_norm_out)
    ```

### 2.4 实例归一化Instance Normalization（IN）<sup>[7]</sup>
#### 2.4.1 提出背景
对于图像风格迁移这类的注重每个像素的细粒度任务来说，每个样本的每个像素点的信息都是非常重要的，于是像BN这种每个批量的所有样本都做归一化的算法就不太适用了，因为BN计算归一化统计量时考虑了一个批量中所有图片的内容，从而造成了每个样本独特细节的丢失。同理对于LN这类需要考虑一个样本所有通道的算法来说可能忽略了不同通道的差异，也不太适用于图像风格迁移这类应用。

#### 2.4.2 概念及公式
对于 <img src="https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{N\times{C}\times{H}\times{W}}"> ，IN 对每个样本的 H、W 维度的数据求均值和标准差，保留 N 、C 维度，也就是说，它只在 channel 内部求均值和标准差，其公式为：

<div align="center"><img src="https://latex.codecogs.com/svg.latex?u_{nc}(x)=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}x_{nchw}\\\sigma_{nc}(x)=\sqrt{\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}\left(x_{nchw}-\mu_{nc}(x)\right)^{2}+\epsilon}"></div>

![image-20220220102433261](https://raw.githubusercontent.com/RangeKing/Cloud-Image/main/img/202202201024539.png)

<div align=center>图4: CNN中IN示意图</div><br></br>

IN 求均值时，相当于把一张照片中所有像素值加起来，再除以该张照片总像素数：H×W，即求每张照片的平均像素值，求标准差时也是同理。

#### 2.4.3 IN代码实践

```python
class paddle.nn.InstanceNorm2D(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None, data_format="NCHW", name=None)
```

该接口用于构建 `InstanceNorm2D` 类的一个可调用对象，具体参数详情参考[InstanceNorm2D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm2D_cn.html#instancenorm2d)。

- 重要参数含义：

    - `num_features` (int) - 指明输入 Tensor 的通道数量。

- 输入输出形状：

    - input: 形状为（批大小，通道数，高度，宽度）的4-D Tensor。

    - output: 和输入形状一样。

- 代码示例如下：

    ```python
    # Instance Norm
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm2D_cn.html#instancenorm2d
    instance_norm = paddle.nn.InstanceNorm2D(2)
    instance_norm_out = instance_norm(x)
    
    print('---------------[Instance Norm Output]---------------\n',instance_norm_out)
    ```

### 2.5 组归一化Group Normalization（GN）<sup>[8]</sup>
#### 2.5.1 提出背景
**Group Normalization (GN) 适用于占用显存比较大的任务，例如图像分割**。对这类任务，可能 batchsize 只能是个位数，再大显存就不够用了。而当 batchsize 是个位数时，BN 的表现很差，因为没办法通过几个样本的数据量，来近似总体的均值和标准差。**GN 也是独立于 batch 的，它是 LN 和 IN 的折中**。

#### 2.5.2 概念及公式
<div align="center"><img src="https://latex.codecogs.com/svg.latex?u_{n g}(x)=\frac{1}{(C/G) H W} \sum_{c=gC/G}^{(g+1)C/G} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n c h w} \\
\sigma_{n g}(x)=\sqrt{\frac{1}{(C/G) H W} \sum_{c=gC/G}^{(g+1)C/G} \sum_{h=1}^{H} \sum_{w=1}^{W}\left(x_{n c h w}-\mu_{n g}(x)\right)^{2}+\epsilon}"></div>

![image-20220220102507902](https://raw.githubusercontent.com/RangeKing/Cloud-Image/main/img/202202201025998.png)</div>

<div align=center>图5: CNN中GN的示意图</div><br></br>

继续用相册类比，GN 相当于把一本 C 页的相册平均分成 G 份，每份成为有 C/G 页的小册子，求每个小册子的平均像素值和标准差。

#### 2.5.3 GN代码实践

```python
class paddle.nn.GroupNorm(num_groups, num_channels, epsilon=1e-05, weight_attr=None, bias_attr=None, data_layout='NCHW, 'name=None)
```

该接口用于构建 `GroupNorm` 类的一个可调用对象，具体参数详情参考[GroupNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GroupNorm_cn.html#groupnorm)。

- 重要参数含义：

    - `num_groups` (int) - 从通道中分离出来的 group 的数目。
    - `num_channels` (int) - 输入的通道数。

- 输入输出形状：

    - input: 形状为(批大小, 通道数, *) 的Tensor。

    - output: 和输入形状一样。

- 代码示例如下：

    ```python
    # Group Norm
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GroupNorm_cn.html#groupnorm
    group_norm = paddle.nn.GroupNorm(num_channels=2, num_groups=2)
    group_norm_out = group_norm(x)
    
    print('---------------[Group Norm Output]---------------\n',group_norm_out)
    ```

    

## 3 总结

### 3.1 对比


如果把 <img src="https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^{N\times{C}\times{H}\times{W}}">类比为一叠相册，这叠相册总共有 N 本，每本有 C 页，每页照片的尺寸[长×宽]为[H×W]。


| 方法         | 类比                                                         | 应用场景                                    |
| :----------- | :----------------------------------------------------------- | :------------------------------------------ |
| BatchNorm    | BN 求均值时，相当于把这些相册按页码一一对应地加起来，再除以这些照片的像素总数：N×H×W。BN 求均值是一个“跨相册求平均”的操作 | Batch Size较大的CNN、MLP                    |
| LayerNorm    | LN 求均值时， 相当于把每一本相册的所有像素值加起来，再除以这本相册的像素总数：C×H×W，即求整本相册的平均像素值 | RNN<br />Transformer                        |
| InstanceNorm | IN 求均值时，相当于把一页相册中所有像素值加起来，再除以该页的总像素数：H×W，即求单张照片的平均像素值 | GAN-风格迁移等细粒度任务                    |
| GroupNorm    | GN 求均值时，相当于把一本 C 页的相册平均分成 G 份，每份成为有 C/G 页的小册子，对这个 C/G 页的小册子，求每个小册子的平均像素值 | Batch Size较小的CNN、MLP（如Batch Size<16） |

此外，还需要注意它们的映射参数γ和β的区别：对于 BN，IN，GN， 其γ和β都是维度等于通道数 C 的向量。而对于 LN，其γ和β都是维度等于 `normalized_shape` 的矩阵。
最后，BN和IN 可以设置参数：`momentum` 和 `track_running_stats` 来获得在全局数据上更准确的 `running mean` 和 `running std`。而 LN 和 GN 只能计算当前 batch 内数据的真实均值和标准差。

### 3.2 归一化层在网络中的位置
#### 3.2.1 BN层位置
以经典网络ResNet为例，BN层是放置在3×3卷积层之后，激活函数层之前。简单构造如下所示。

```python
class BasicBlock(nn.Layer):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2D(in_dim, out_dim, 3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_dim, out_dim, 3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_dim)

        if in_dim != out_dim or stride == 2:
            self.downsample = nn.Sequential(*[
                nn.Conv2D(in_dim, out_dim, 1, stride=stride),
                nn.BatchNorm2D(out_dim)])
        else:
            self.downsample = Identity()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(h)
        x = x + identity
        x = self.relu(x)
        return x
```
PaddleClas中关于ResNet的具体代码详见 [PaddleClas Github 官方仓库](https://github.com/PaddlePaddle/PaddleClas/blob/1358e3f647e12b9ee6c5d6450291983b2d5ac382/ppcls/arch/backbone/legendary_models/resnet.py#L107-L147)。

有兴趣的还可以阅读一下2018年的一篇论文 [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/abs/1801.05134)。


#### 3.2.2 LN层位置
最近大火的Transformer中就用到LN。谷歌[ViT](https://arxiv.org/abs/2010.11929)的论文中LN层是放置在多头自注意力模块和残差连接相加之后（如图6(a)所示），这种做法被称为 PostNorm。

之后有[论文](https://arxiv.org/abs/2002.04745)改变LN层的位置并做了相关实验，发现LN层防止在多头自注意力模块之前整个网络训练成功率会较高一点。而这种做法也被称为 PreNorm。拿下 ICCV 2021 最佳论文奖的Swin Transformer，使用的也是PreNorm，具体代码可见[Swin Transformer Github官方仓库](
https://github.com/microsoft/Swin-Transformer/blob/5d2aede42b4b12cb0e7a2448b58820aeda604426/models/swin_transformer.py#L233-L270)。

<div align=center><img src="https://raw.githubusercontent.com/RangeKing/Cloud-Image/main/img/202202262055529.png"></div>

<div align=center>图6: Transformer中的 (a)PostNorm (b)PreNorm</div><br></br>

### 3.3 Paddle中已实现的归一化方法

不完全归纳

| 方法              | Paddle API文档                                               |
| ----------------- | ------------------------------------------------------------ |
| BatchNorm         | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/BatchNorm_cn.html#batchnorm |
| LayerNorm         | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#layernorm |
| InstanceNorm      | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/InstanceNorm2D_cn.html#instancenorm2d |
| GroupNorm         | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/GroupNorm_cn.html#groupnorm |
| SpectralNorm      | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/SpectralNorm_cn.html#spectralnorm |
| LocalResponseNorm | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LocalResponseNorm_cn.html#localresponsenorm |
| WeightNorm        | https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/utils/weight_norm_cn.html#weight-norm |




## 4 扩展阅读

> 1. **Weight Normalization**, 2016. [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks.](https://arxiv.org/abs/1602.07868)
> 2. **Spectral Normalization**, 2018. [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).
> 3. **Switchable Normalization**, 2019. [Differentiable Learning-to-Normalize via Switchable Normalization.](https://arxiv.org/abs/1806.10779)
> 4. **Dynamic Normalization**, 2021. [Dynamic Normalization.](https://arxiv.org/abs/2101.06073)
> 5. **Cross Normalization**, 2021. [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://arxiv.org/abs/2102.02811).



## Reference

> [1] PaddleEdu. [归一化基础知识点](https://github.com/PaddlePaddle/awesome-DeepLearning/blob/master/docs/tutorials/deep_learning/normalization/basic_normalization.md). Github: PaddlePaddle/awesome-DeepLearning. 2021.
>
> [2] P Bruce, A Bruce. Practical statistics for data scientists: 50 essential concepts.  2017.
>
> [3] Dong. [BN、LN、IN、GN的简介](https://zhuanlan.zhihu.com/p/91965772). 知乎. 2019.
>
> [4] Ioffe S, Szegedy C. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). International conference on machine learning. PMLR, 2015: 448-456.
>
> [5] Santurkar S, Tsipras D, Ilyas A, et al. [How does batch normalization help optimization?](https://arxiv.org/abs/1805.11604). Advances in neural information processing systems. 2018, 31. 
>
> [6]  Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/abs/1607.06450). arXiv preprint. 2016.
>
> [7] Ulyanov D, Vedaldi A, Lempitsky V. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022). arXiv preprint. 2016.
>
> [8] Wu Y, He K. [Group Normalization](https://arxiv.org/abs/1803.08494). Proceedings of the European conference on computer vision (ECCV). 2018: 3-19.
