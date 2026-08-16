#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：down_dataset.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/8/15 17:01 
@Description： 
'''

if __name__ == '__main__':
    import kagglehub

    # Download latest version
    path = kagglehub.competition_download('digit-recognizer')

    print("Path to competition files:", path)