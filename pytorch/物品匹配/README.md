# 多角度物品图像向量检索识别系统

## 1. 项目目标
新品/换包装后，怎么做到低成本、快速上线，同时识别稳定
料上新与换包装太快，领导要求换了包装后用户识别要立马识别到，所以我们采用这套方案，每个饮品我们都会拍十几二十几张图片甚至几十张图片然后形成特征库，然后用户拍照后，对其进行识别，然后匹配出饮品
构建一个基于图像特征向量和向量数据库的物品识别系统。

系统不要求每增加一种物品就重新训练一个分类模型，而是通过为已有物品建立多角度图像特征库，在识别阶段将待识别图片转换为特征向量，并通过向量相似度检索判断其最可能对应的物品。

---
## 流程

项目背景：由于饮品上新和包装更换频繁，如果采用传统分类模型，每次新增饮品或更换包装都需要重新训练和发布模型，成本较高。因此采用“饮品特征库 + 向量检索”的方式，使新品或新包装在采集图片并建立特征后即可快速支持识别。

整体流程：

用户拍摄原图
      ↓
① 饮品目标检测
      ↓
得到 N 个饮品 Box
      ↓
Crop / ROI
      ↓
② 饮品视觉特征提取
      ↓
Embedding
      ↓
③ 向量检索
      ↓
④ 候选饮品 Top-K
      ↓
⑤ 匹配决策
      ↓
饮品 ID / Unknown

因此整个系统可以概括为：

## 代码结构
分成四个模块：
1. Detection： 输入 原始图片，输出box
2. Feature Extraction ： 输入crop，输出Embeddings 即这个饮品张什么样子
3. Vector Retrieval： 从已有饮品中找到最相似筛选
4. Decision： 决策层
## 第一阶段
“把图片中所有可能属于饮品的目标框出来。”
使用RT-DETR模型
## Feature Extractor
这个模块的模型要学习的是：什么样的视觉特征，能够代表这瓶饮品的身份？

采用 预训练模型-提取Embedding-建立milvus特征库-实际测试的流程

预训练模型选项：DINOv2 CLIP ViT ResNet
### 预训练模型怎么选？
Embedding不取分类logits、也不用最后一层卷积的铽征途，而是取模型经过全局聚合后的、用于表达整个图视觉信息的特征，然后再做L2Normalize
一下是三种模型的区别：
1. ResNet
主要学习图像分类，最适合当BaseLine。但是不能全部用，因为ResNet更关心商品类别比如猫、狗、羊，我们想要的区别是可口可乐、百事可乐、百事无糖这种。
这种属于细粒度商品识别。
2. CLIP
CLIP它不仅识别图像，还会识别文本。饮品包装上都有一些简单文字比如名称。这个模型可以学习到大量的 视觉+语义 的信息，对商品识别很有价值。

商品来了以后，先将图像Embedding、然后再将文本Embedding。

3. DINOv2 
从图像本身学习通用视觉表征，这个模型更关注：形状、纹理、结构、局部视觉特征、整体视觉特征、物体之间的视觉关系

这个看起来和ResNet很像

所以最后特征提取器使用三个模型（ResNet、CLIP、DINOv2）然后这三个模型输出特征进行L2 Normalize 最后存储Milvus中。

ResNet取全局平均池化后面的那一层，有空间尺寸数据，也有特征。（因为gap后空间信息都会被压缩到1×1）


剩下的ViT / DINOv2，DINOv2是transformer架构，

Transformer 会产生：cls是全局token，patch token是局部特征。
CLS token（人为加入的 Token 序列）
+
Patch tokens
这里DINOv2就是用的 CLS / global representation 作为这个图的Embedding。CLS讲的是整个图片整体什么样，Patch token描述的是这个局部区域怎么样


所以最后这三个模型分别变成了：
ResNet
```shell
ResNet50
 ↓
Backbone
 ↓
Global Average Pooling
 ↓
2048-d
 ↓
L2 Normalize
```
CLIP
```shell
Image Encoder
 ↓
Image Embedding
 ↓
L2 Normalize
```
DINOv2
```shell
DINOv2
 ↓
Global / CLS Representation
 ↓
L2 Normalize
```


### 可能遇到的疑问

需要让模型学习到：

```text
同一种物品：
不同角度 / 不同姿态 / 合理光照变化
        ↓
特征向量距离较近
```

同时：

```text
不同物品
        ↓
特征向量距离较远
```

因此后续需要重点研究：

* 使用什么特征提取模型
* 使用 ImageNet 预训练特征是否足够
* 是否需要针对物品数据进行微调
* 如何处理不同角度
* 如何处理背景变化
* 如何处理光照变化
* 如何确定相似度阈值
* 如何处理数据库中不存在的物品

---
