#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：benchmark.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/19 17:51 
@Description： 
'''
if __name__ == '__main__':
    from ultralytics.utils.benchmarks import benchmark

    # Benchmark on GPU
    benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, device='mps')

    # Benchmark specific export format
    benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, format="onnx")