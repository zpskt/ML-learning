#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：val.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/18 00:48 
@Description： 
'''
if __name__ == '__main__':
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.pt")  # load an official model
    model = YOLO("/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/runs/detect/train-5/weights/best.pt")  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list containing mAP50-95 for each category
    metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
