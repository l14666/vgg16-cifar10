# VGG16 on CIFAR-10

基于PyTorch实现的VGG16模型，用于CIFAR-10数据集分类任务。

## 项目结构

```
.
├── models/
│   └── vgg16.py         # VGG16模型定义
├── utils/
│   └── data_loader.py   # 数据加载和处理
├── train.py             # 训练脚本
├── main.py              # 主程序入口
├── checkpoints/         # 模型检查点保存目录
└── README.md            # 项目说明
```

## 实验目的

本实验旨在通过在CIFAR-10数据集上训练VGG16模型，探究不同超参数对模型性能的影响，包括：

- 权重初始化方法
- 激活函数类型
- 卷积核大小
- feature map数量
- 每层及全连接层数量
- 神经元数量
- 损失函数
- 优化器选择
- 学习率

通过实验，分析各超参数模型性能和训练效率的作用，并总结神经网络结构在图像分类任务中的特性与规律。

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib
- numpy

## 安装依赖

```bash
pip install torch torchvision matplotlib numpy
```

## 使用方法

### 训练模型

```bash
python main.py --mode train --batch-size 128 --epochs 100 --lr 0.001 --optimizer sgd
```

### 测试模型

```bash
python main.py --mode test --checkpoint best_model.pth
```

### 可视化预测结果

```bash
python main.py --mode visualize --checkpoint best_model.pth
```

### 使用TensorBoard查看训练过程

训练过程中，损失(loss)和准确率(accuracy)会自动记录到TensorBoard中。训练完成后，可以使用以下命令启动TensorBoard服务器：

```bash
tensorboard --logdir=runs
```

然后在浏览器中打开 http://localhost:6006 查看训练曲线。

TensorBoard记录的指标包括：
- 训练损失(Loss/train)
- 测试损失(Loss/test)
- 训练准确率(Accuracy/train)
- 测试准确率(Accuracy/test)
- 学习率变化(Learning_rate)

不同超参数配置的训练结果会保存在不同的目录中，便于比较不同配置的性能。

## 参数说明

### 运行模式

- `--mode`: 运行模式，可选 'train', 'test', 'visualize'，默认为 'train'

### 数据集参数

- `--batch-size`: 批次大小，默认为 128
- `--num-workers`: 数据加载的工作线程数，默认为 4
- `--augment`: 是否使用数据增强，默认为 True

### 模型参数

- `--activation`: 激活函数类型，可选 'relu', 'leaky_relu', 'elu'，默认为 'relu'
- `--kernel-size`: 卷积核大小，默认为 3
- `--feature-maps`: 初始特征图数量，默认为 64
- `--fc-layers`: 全连接层数量，默认为 3
- `--neurons`: 全连接层神经元数量，默认为 4096
- `--init-weights`: 是否初始化权重，默认为 True

### 训练参数

- `--epochs`: 训练轮数，默认为 100
- `--lr`: 学习率，默认为 0.01
- `--momentum`: SGD动量，默认为 0.9
- `--weight-decay`: 权重衰减，默认为 5e-4
- `--optimizer`: 优化器，可选 'sgd', 'adam', 'rmsprop'，默认为 'sgd'
- `--loss`: 损失函数，可选 'cross_entropy', 'nll', 'focal'，默认为 'cross_entropy'
  - `cross_entropy`: 交叉熵损失函数，适用于多分类问题，使用 `nn.CrossEntropyLoss()`
  - `nll`: 负对数似然损失函数，需要在模型最后一层添加LogSoftmax，使用 `nn.NLLLoss()`
  - `focal`: Focal Loss，用于处理类别不平衡问题，自定义实现

  **注意**: train.py中特别处理了NLLLoss的预测逻辑，其他损失函数则使用默认预测方式。

### 其他参数

- `--no-cuda`: 禁用CUDA训练，默认为 False
- `--seed`: 随机种子，默认为 1
- `--save-dir`: 检查点保存目录，默认为 'checkpoints'
- `--checkpoint`: 检查点文件名，默认为 'best_model.pth'

## 实验示例

### 不同激活函数的比较

```bash
# ReLU激活函数
python main.py --mode train --activation relu --epochs 100

# LeakyReLU激活函数
python main.py --mode train --activation leaky_relu --epochs 100

# ELU激活函数
python main.py --mode train --activation elu --epochs 100
```

### 不同优化器的比较

```bash
# SGD优化器
python main.py --mode train --optimizer sgd --lr 0.001 --epochs 100

# Adam优化器
python main.py --mode train --optimizer adam --lr 0.001 --epochs 100

# RMSprop优化器
python main.py --mode train --optimizer rmsprop --lr 0.001 --epochs 100
```

### 不同损失函数的比较

```bash
# 交叉熵损失函数
python main.py --mode train --loss cross_entropy --epochs 100

# 负对数似然损失函数
python main.py --mode train --loss nll --epochs 100

# Focal Loss
python main.py --mode train --loss focal --epochs 100
```

### 不同卷积核大小的比较

```bash
# 卷积核大小为3
python main.py --mode train --kernel-size 3 --epochs 100

# 卷积核大小为5
python main.py --mode train --kernel-size 5 --epochs 100
```

### 不同特征图数量的比较

```bash
# 初始特征图数量为32
python main.py --mode train --feature-maps 32 --epochs 100

# 初始特征图数量为64
python main.py --mode train --feature-maps 64 --epochs 100

# 初始特征图数量为128
python main.py --mode train --feature-maps 128 --epochs 100
