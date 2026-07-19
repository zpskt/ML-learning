#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：export_mode2onnx.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/19 11:16 
@Description： 
'''
if __name__ == '__main__':
    from ultralytics import YOLO

    model_path = '/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/runs/detect/train-8/weights/best.pt'
    # Load a pretrained YOLO26n model
    model = YOLO(model_path)

    # Export the model to ONNX format
    model.export(format="onnx")  # creates 'yolo26n.onnx'

    # Export an INT8-quantized ONNX model with calibration data
    model.export(format="coreml", quantize=8)  # creates 'yolo26n.mlpackage'
