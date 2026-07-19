#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：track.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/19 15:23 
@Description： 
'''
if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    # Default tracker (BoT-SORT)
    results = model.track(source="/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/datasets/VID_20260719_174227.mp4", show=True)

    # Switch to ByteTrack
    results = model.track(source="/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/datasets/VID_20260719_174227.mp4", show=True, tracker="bytetrack.yaml")
