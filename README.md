# ML-learning 计算机视觉与机器学习基础项目

## 项目介绍

本项目是一个计算机视觉和机器学习的基础代码集合，适合初学者快速上手并进行演示。项目包含了使用OpenCV进行图像处理、基于PyTorch的深度学习模型实现、YOLO模型的食物识别系统以及Kaggle竞赛项目示例。

### 目标用户

- 初学者和对计算机视觉、机器学习感兴趣的学习者
- 希望通过实践代码理解机器学习概念的开发者
- 需要参考实现基础计算机视觉功能的工程师

## 项目结构

```
ML-learning/
├── MyOpneCV/                  # OpenCV基础操作示例
├── document/                  # 相关文档
├── food_recognition/          # 基于YOLO的食物识别系统
├── kaggle/                    # Kaggle竞赛项目示例
│   └── Predicting-Optimal-Fertilizers/
├── pytorch/                   # PyTorch基础教程
└── README.md                  # 项目说明文档
```

## 各模块详细介绍

### 1. MyOpneCV - OpenCV基础操作

包含了一系列OpenCV的基础操作示例：

- 基本图像操作（读取、显示、保存）
- 视频处理（读取摄像头、保存视频）
- 图像处理技术（直方图、图像清晰化）
- 人脸检测（基于Haar特征）
- 腐蚀操作等形态学处理

### 2. pytorch - PyTorch基础教程

PyTorch学习代码，从基础到进阶：

- 张量基础操作和运算
- 自动微分系统
- 线性回归和逻辑回归实现
- 多层感知器(MLP)实现

### 3. food_recognition - 食物识别系统

基于YOLOv8的食物识别系统，具有以下特性：

- 数据预处理模块
- 模型训练和重训练模块
- API服务模块
- 实时摄像头检测功能

#### 目录结构

```
food_recognition/
├── data/                      # 数据目录
│   └── train/
│       └── food/
├── src/                       # 源代码目录
│   ├── api/                   # API服务模块
│   ├── train/                 # 初始训练模块
│   ├── retrain/               # 重训练模块
│   └── data_processing/       # 数据处理模块
├── models/                    # 模型保存目录
├── requirements.txt           # 项目依赖
└── README.md                  # 模块说明文档
```

### 4. kaggle/Predicting-Optimal-Fertilizers - Kaggle竞赛项目

基于Kaggle竞赛"Predicting Optimal Fertilizers"的完整解决方案：

- 数据分析和预处理
- 特征工程
- 多种模型训练和比较
- 结果生成和提交

## 环境安装

### 基础环境要求

- Python 3.7+
- pip包管理器

### 安装依赖

```bash
pip install -r requirements.txt
```

### 各模块特定依赖

#### food_recognition模块

```bash
# 进入food_recognition目录
cd food_recognition

# 安装依赖
pip install -r requirements.txt
```

主要依赖包括:
- ultralytics (YOLOv8)
- flask
- opencv-python

#### 创建conda环境（推荐）

```bash
conda create -n kaggle --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
conda activate food
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install pickle numpy matplotlib -i https://mirrors.aliyun.com/pypi/simple/
```

## 使用说明

### OpenCV示例运行

```bash
cd MyOpneCV
python 01-opencv常用基本操作.py
```

### PyTorch教程运行

```bash
cd pytorch
python 02-线性回归.py
```

### 食物识别系统

#### 数据准备

将食材图片按照以下结构存放：

```
data/train/food/
├── ingredient1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── ingredient2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ...
```

#### 训练模型

```bash
cd food_recognition/src/train
python train_model.py
```

#### 启动API服务

```bash
cd food_recognition/src/api
python app.py
```

### Kaggle项目运行

```bash
cd kaggle/Predicting-Optimal-Fertilizers
python data_analysis.py
python train_model_v1.py
```

## 技术栈

- OpenCV - 计算机视觉库
- PyTorch - 深度学习框架
- YOLOv8 - 目标检测模型
- Flask - Web API框架
- Scikit-learn - 机器学习库

## 项目特点

1. **模块化设计** - 每个功能模块独立，便于学习和使用
2. **完整示例** - 从数据处理到模型部署的完整流程
3. **详细注释** - 代码中包含详细注释，适合初学者理解
4. **实践导向** - 基于真实项目和竞赛数据

## 贡献

欢迎提交Issue和Pull Request来改进项目。