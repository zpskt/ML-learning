# 食材识别

本项目是一个基于YOLO的食材识别系统，可以识别视频中出现的食材。

作者: zhangpeng
时间: 2025-08-28

## 目录结构

```
food_recognition/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── food_dataset.yaml      # 数据集配置文件
├── src/
│   ├── api/                   # API服务模块
│   ├── train/                 # 初始训练模块
│   ├── retrain/               # 重训练模块
│   ├── data_processing/       # 数据处理模块
│   ├── events/                # 事件处理模块
│   ├── config/                # 配置管理模块
│   ├── exceptions/            # 异常处理模块
│   ├── utils/                 # 工具模块
│   └── tests/                 # 测试模块
├── models/                    # 模型保存目录
├── docs/                      # 文档目录
├── logs/                      # 日志目录
├── config.json                # 全局配置文件
├── requirements.txt           # 项目依赖
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

数据准备格式：
将食材图片按照以下结构存放：

```
data/food/
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

### 2. 初始训练模块

用于从头开始训练食材识别模型。

```bash
cd src/train
python train_model.py --data_config ../../data/food_dataset.yaml --epochs 100 --save_dir ../../models
```

### 3. 重训练模块

在已有模型基础上进行增量训练。

```bash
cd src/retrain
python retrain_model.py --model_path ../../models/food_model/weights/best.pt --data_config ../../data/food_dataset.yaml --epochs 50 --save_dir ../../models
```

### 4. API服务模块

提供视频食材识别的API服务。

```bash
cd src/api
python app.py
```

启动后访问 `http://localhost:5000` 查看API信息。

实时摄像头检测：
```bash
cd src/api
python camera_detector.py --model_path ../../models/food_model/weights/best.pt
```

按 'q' 键退出摄像头检测。

### 5. 事件处理模块

当识别到食材时触发事件，支持多种处理方式：
- 日志记录
- 语音播报
- 外部接口调用
- Kafka消息推送

该模块采用解耦设计，可以轻松扩展新的事件处理方式。

事件处理由配置文件控制，默认只启用了日志记录事件监听器。

### 6. 配置管理模块

项目使用统一的配置管理机制，所有模块通过读取配置决定自己的行为。

配置文件位于项目根目录的 [config.json](file:///Users/zhangpeng/Desktop/zpskt/ML-learning/food_recognition/config.json) 文件中，可以控制以下功能：

```json
{
  "event_handlers": {
    "log": true,     // 启用日志记录
    "tts": false,    // 启用语音播报
    "api": false,    // 启用外部API调用
    "kafka": false   // 启用Kafka推送
  },
  "tts": {
    "enabled": false,
    "rate": 200,
    "voice": "default"
  },
  "api": {
    "enabled": false,
    "endpoint": ""
  },
  "kafka": {
    "enabled": false,
    "bootstrap_servers": ["localhost:9092"],
    "topic": "food-detection-events"
  },
  "logging": {
    "enabled": true,              // 启用日志记录到文件
    "file": "logs/food_detection.log",  // 日志文件路径
    "level": "INFO",              // 日志级别
    "max_bytes": 10485760,        // 单个日志文件最大字节数 (10MB)
    "backup_count": 5             // 保留的备份日志文件数量
  }
}
```

要启用特定功能，只需修改配置文件中对应项为 `true` 并填写相关配置即可。

运行示例脚本查看效果：
```bash
cd src/events
python run_example.py
```

### 7. 异常处理模块

项目定义了统一的异常类型，所有自定义异常都继承自 `FoodRecognitionException` 基类。

### 8. 工具模块

包含日志配置等通用工具函数。

### 9. 测试模块

包含单元测试代码。

## 文档

- [用户手册](./docs/user_manual.md) - 详细使用说明
- [开发指南](./docs/development_guide.md) - 开发和扩展指南

## 数据准备

将食材图片按照以下结构存放：

```
data/food/
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

## Flask API接口文档

### 1. 首页

**URL**: `GET /`

**描述**: 返回API服务信息和可用接口列表

**响应示例**:
```json
{
  "message": "食材识别API服务",
  "endpoints": {
    "/load_model": "加载模型",
    "/detect": "检测视频中的食材",
    "/detect_image": "检测图片中的食材"
  }
}
```

### 2. 加载模型接口

**URL**: `POST /load_model`

**描述**: 加载指定路径的模型文件

**请求参数**:
```json
{
  "model_path": "模型文件路径，如: ../../models/food_model/weights/best.pt"
}
```

**响应示例**:
```json
{
  "message": "模型加载成功"
}
```

### 3. 图片食材检测接口

**URL**: `POST /detect_image`

**描述**: 检测图片中的食材

**请求参数**:
```json
{
  "image_path": "图片文件路径",
  "conf_threshold": 0.5  // 可选，置信度阈值，默认为0.5
}
```

**响应示例**:
```json
{
  "message": "检测完成",
  "ingredients": ["土豆", "胡萝卜"]
}
```

### 4. 视频食材检测接口

**URL**: `POST /detect`

**描述**: 检测视频中的食材

**请求参数**:
```json
{
  "video_path": "视频文件路径",
  "conf_threshold": 0.5  // 可选，置信度阈值，默认为0.5
}
```

**响应示例**:
```json
{
  "message": "检测完成",
  "ingredients": ["土豆", "胡萝卜", "青椒"]
}
```