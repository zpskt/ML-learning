#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：kaggle_down.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/18 20:48 
@Description： 
'''

if __name__ == '__main__':
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("denuwanwijesinghe/vegetable-object-detection")

    print("Path to dataset files:", path)
