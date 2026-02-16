#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/2/16 09:50
# @Author  : zhangpeng /zpskt
# @File    : test_local_model.py
# @Software: PyCharm


if __name__ == '__main__':
    from paddleocr import TextRecognition

    model = TextRecognition(model_name="PP-OCRv5_server_rec")
    output = model.predict(input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png", batch_size=1)
    for res in output:
        res.print()
        res.save_to_img(save_path="./output/")
        res.save_to_json(save_path="./output/res.json")
    from paddleocr import PaddleOCR

    pipeline = PaddleOCR()
    pipeline.export_paddlex_config_to_yaml("PaddleOCR.yaml")