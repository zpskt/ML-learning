#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/2/8 20:44
# @Author  : zhangpeng /zpskt
# @File    : test.py
# @Software: PyCharm
import cv2
#2使用ocr
from paddleocr import PaddleOCR


if __name__ == '__main__':
    from paddleocr import PaddleOCR

    # 初始化 PaddleOCR 实例
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # 对示例图像执行 OCR 推理
    result = ocr.predict(
        input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

    # 可视化结果并保存 json 结果
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")