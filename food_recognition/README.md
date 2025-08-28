# 食材识别

本项目是一个基于YOLO的食材识别系统，可以识别视频中出现的食材。

作者: zhangpeng
时间: 2025-08-28

## 目录结构

```
food_recognition/
├── data/
│   └── train/
│       └── food/
├── src/
│   ├── api/           # API服务模块
│   ├── train/         # 初始训练模块
│   ├── retrain/       # 重训练模块
│   └── data_processing/  # 数据处理模块
├── models/            # 模型保存目录
├── requirements.txt   # 项目依赖
└── README.md
```

## 创建环境

```shell
conda create -n food --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9
```

安装依赖

```shell
conda activate food
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 项目模块说明

### 1. 数据处理模块

处理原始图片数据，创建标签文件，并将数据拆分为训练集和验证集。

```bash
cd src/data_processing

python prepare_data.py
```

### 2. 初始训练模块

用于从头开始训练食材识别模型。

```bash
cd src/train
python train_model.py --data_dir ../../data/yolo_dataset --epochs 100 --save_dir ../../models
```

### 3. 重训练模块

在已有模型基础上进行增量训练。

```bash
cd src/retrain
python retrain_model.py --model_path ../../models/food_model/weights/best.pt --data_dir ../../data/yolo_dataset --epochs 50 --save_dir ../../models
```

### 4. API服务模块

提供视频食材识别的API服务。

```bash
cd src/api
python app.py
```

启动后访问 `http://localhost:5000` 查看API信息。

## 数据准备

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

每个子文件夹代表一种食材，文件夹名为食材名称。

## YOLO标签格式说明

在YOLO格式中，每个图像对应的标签文件（.txt）包含边界框的信息，每行代表一个检测对象，格式为：
```
<object_class> <x_center> <y_center> <width> <height>
```

其中：
- `object_class`：对象类别ID（整数）
- `x_center`：边界框中心点的x坐标（相对于图像宽度的比例，0-1之间）
- `y_center`：边界框中心点的y坐标（相对于图像高度的比例，0-1之间）
- `width`：边界框的宽度（相对于图像宽度的比例，0-1之间）
- `height`：边界框的高度（相对于图像高度的比例，0-1之间）

在当前食材分类任务中，使用占位标签：
```
0 0.5 0.5 1.0 1.0
```

表示：
- `0`：类别ID（例如0代表土豆，1代表西红柿等）
- `0.5 0.5`：边界框中心点位于图像中心（x=50%，y=50%）
- `1.0 1.0`：边界框宽度和高度都占满整个图像（100%）

这种设计适用于图像分类任务，假设整个图像都是该类别的食材，在YOLO模型训练中是合理的。