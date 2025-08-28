# 用户手册

## 简介

食材识别系统是一个基于YOLO的计算机视觉项目，能够识别视频中出现的食材。

## 安装指南

### 环境要求

- Python 3.7+
- pip包管理工具

### 安装步骤

1. 克隆项目代码
```bash
git clone <项目地址>
cd food_recognition
```

2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 安装可选依赖

根据需要安装额外功能的依赖包：

```bash
# 语音播报功能
pip install pyttsx3

# Kafka推送功能
pip install kafka-python

# 外部API调用
pip install requests
```

## 数据准备

### 数据结构

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

### 数据处理

处理原始图片数据，创建标签文件，并将数据拆分为训练集和验证集。

```bash
cd src/data_processing
python prepare_data.py
```

## 模型训练

### 初始训练

用于从头开始训练食材识别模型。

```bash
cd src/train
python train_model.py --data_config ../../data/food_dataset.yaml --epochs 100 --save_dir ../../models
```

### 重训练

在已有模型基础上进行增量训练。

```bash
cd src/retrain
python retrain_model.py --model_path ../../models/food_model/weights/best.pt --data_config ../../data/food_dataset.yaml --epochs 50 --save_dir ../../models
```

## 使用方法

### API服务

提供视频食材识别的API服务。

```bash
cd src/api
python app.py
```

启动后访问 `http://localhost:5000` 查看API信息。

### 实时摄像头检测

```bash
cd src/api
python camera_detector.py --model_path ../../models/food_model/weights/best.pt
```

按 'q' 键退出摄像头检测。

## 配置说明

项目使用统一的配置管理机制，配置文件位于项目根目录的 `config.json` 文件中。

### 启用TTS语音播报

修改 `config.json` 文件：

```json
{
  "event_handlers": {
    "log": true,
    "tts": true,
    "api": false,
    "kafka": false
  },
  "tts": {
    "enabled": true,
    "rate": 200,
    "voice": "default"
  }
}
```

### 启用外部API调用

修改 `config.json` 文件：

```json
{
  "event_handlers": {
    "log": true,
    "tts": false,
    "api": true,
    "kafka": false
  },
  "api": {
    "enabled": true,
    "endpoint": "http://example.com/api/food-detection"
  }
}
```

### 启用Kafka推送

修改 `config.json` 文件：

```json
{
  "event_handlers": {
    "log": true,
    "tts": false,
    "api": false,
    "kafka": true
  },
  "kafka": {
    "enabled": true,
    "bootstrap_servers": ["localhost:9092"],
    "topic": "food-detection-events"
  }
}
```

### 配置日志记录

日志记录默认是启用的，可以通过修改 `config.json` 文件中的 `logging` 部分来配置：

```json
{
  "logging": {
    "enabled": true,
    "file": "logs/food_detection.log",
    "level": "INFO",
    "max_bytes": 10485760,
    "backup_count": 5
  }
}
```

配置项说明：
- `enabled`: 是否启用日志记录到文件
- `file`: 日志文件路径
- `level`: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `max_bytes`: 单个日志文件的最大字节数
- `backup_count`: 保留的备份日志文件数量

## API接口文档

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

## 故障排除

### 常见问题

1. **无法打开摄像头**
   - 检查摄像头是否被其他程序占用
   - 确认摄像头权限设置

2. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件是否存在且未损坏

3. **TTS语音播报无声音**
   - 检查系统音量设置
   - 确认是否安装了pyttsx3库
   - 在macOS上可能需要确认系统语音设置

### 日志查看

系统日志默认输出到控制台和文件。日志文件位置由配置文件中的 `logging.file` 项指定，默认为 `logs/food_detection.log`。

## 联系方式

如有问题，请联系项目维护者。