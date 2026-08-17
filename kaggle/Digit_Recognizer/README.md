#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：README.md.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/8/16 18:00 
@Description： 
'''

---
title: "kaggle-数字图识别-延申知识"
date: 2026-08-17
description: ""
---

笔记地址：https://www.kaggle.com/code/zpskta/digit-recognizer-cnn-mnist
## 环境
```shell
pip install scikit-learn  -i https://mirrors.aliyun.com/pypi/simple/

```

## 延伸问题
### 拿到一个图像识别数据集，你的流程是干什么？
1、审视数据集：尺寸、channel、像素范围、有无异常数据集、类别、类别是否均衡
2、归一化数据集
3、分割训练数据集和验证数据集
4、设计cnn网络
5、开始训练

### 如果数据集发生变更，你要怎么修改
1. 如果图片是彩色的，那么通道变成3个，channel变成3
2. 图片尺寸发生变更，那么fc层的参数数量将会变更（要实时计算最后的空间尺寸*特征channel数量）
3. 如果分类改变，那么需要改变fc2，更改输出映射的类别数量
4. 如果你的识别效果不好，要加层数或者每层的特征识别数量（out_channel数量），那么fc里面的数值要修改
### 拿到陌生的数据集你需要干什么操作？
数据是什么？
│
├── 图片尺寸？
├── Channel？
├── 数据类型？
├── 像素范围？
├── 有没有缺失/异常？
├── 有多少类别？
├── 每类有多少图片？
├── 类别是否平衡？
├── 标签是否可靠？
├── 图片是否存在重复？
├── 图片是否存在明显变形？
├── 训练/验证/测试分布是否一致？
└── 图片中的目标位置、大小、方向是否变化？

### 问题：
Train = 100%
Validation = 96%
而且 validation 已经开始下降。

你会怎么解决？

1. 首先判断为过拟合
2. 在不改变label标签的前提下做数据增强（旋转、拉伸、平移、亮暗程度、噪点）
3. 加dropout层，让前向推导时随机忽略几个特征


