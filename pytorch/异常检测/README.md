# 皮革表面异常检测

## 1. 项目目标

使用深度学习对皮革表面进行异常检测，判断图片是否存在缺陷。

## 2. 数据集

```text
dataset/leather/
├── train/
│   └── good/
└── test/
    ├── good/
    └── defect/
```

训练主要使用正常样本（good），测试使用正常和异常样本。

## 3. 检测流程

```text
输入图片
→ Backbone提取特征
→ 计算Anomaly Score
→ 与Threshold比较
→ Good / Anomaly
```

## 4. 当前结果

目前测试发现 **False Positive = 12**，即 12 张正常图片被误判为异常。

下一步重点分析：

* Threshold 是否合理
* 误判图片的特征
* 正常样本的特征分布
* 模型特征提取效果

## 5. 当前进度

* [x] 数据读取
* [x] 图像预处理
* [x] 模型推理
* [x] Anomaly Score计算
* [x] 异常判断
* [ ] Threshold优化
* [ ] False Positive分析
* [ ] 模型优化
## 问题
1. 怎么解决后期patch膨胀问题
不是“所有 patch 都留着”，而是“只保留能代表正常分布的 patch”。新图片来了以后，只有bank里面没有覆盖过的正常特征才会加进去。
即，每次识别都判断距离，如果距离0.2，0.1远远小于程序阈值，说明是同类的，就不塞入了。