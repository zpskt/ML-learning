#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/16 22:47 
@Description： 
'''
if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.yaml")  # build a new model from YAML
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/datasets/Vegetable-Object-Detection/data.yaml", epochs=100, imgsz=640, device="mps")

