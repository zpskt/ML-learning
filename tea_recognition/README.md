# 简介

## 流程图
```mermaid

flowchart TD
    A[输入茶叶包装图片] --> B(第一步: OCR文字识别)
    B --> C{提取茶叶名称}
    C --> D["核心输出: 茶叶名称 (如'日照绿茶')"]
    
    B --> E(第二步: 类型判断)
    E --> F["方式A: 规则匹配 (性价比高)"]
    E --> G["方式B: 微调小模型 (灵活度高)"]
    F & G --> H["核心输出: 茶叶类型 (如'绿茶')"]
    
    A --> I(第三步: 精准切图)
    I --> J["模型: 专用包装检测模型"]
    J --> K["核心输出: 包装盒坐标<br>与校正后图像"]
    
    D & H & K --> L["最终结果<br>名称+类型+切图"]
    
    L --> M["反馈循环<br>标注错误数据"]
    M --> N["持续优化各环节模型"]
    
```

## 代码结构
tea_recognition_project/
│
├── main.py                      # 主程序入口
├── config.py                    # 配置文件（API密钥、路径等）
│
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── ocr_engine.py           # OCR引擎（PaddleOCR封装）
│   ├── tea_classifier.py       # 茶叶分类器（你的规则库）
│   └── package_detector.py     # 包装盒检测器（YOLO模型）
│
├── models/                      # 存放模型文件
│   ├── paddleocr/              # PaddleOCR模型（会自动下载）
│   └── yolo_tea_package/       # 训练好的YOLO模型
│       ├── weights/best.pt
│       └── data.yaml
│
├── data/                        # 数据目录
│   ├── raw_images/             # 原始茶叶图片
│   ├── annotated/              # 标注后的数据（用于YOLO训练）
│   └── error_cases/            # 识别错误的案例（用于迭代）
│
├── scripts/                     # 实用脚本
│   ├── train_yolo.py           # YOLO训练脚本
│   └── label_images.py         # 图片标注助手脚本
│
└── requirements.txt             # 项目依赖

## 使用教程
### paddleocr
注意事项：要按照官方教程。并且如果numpy报错，换版本1.26.0。
https://github.com/PaddlePaddle/PaddleOCR/blob/main/readme/README_cn.md

```shell
conda create -n tea --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ python=3.9

python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

python -m pip install paddleocr
```