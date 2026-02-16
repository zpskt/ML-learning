#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : ocr_engine.py
# @Software: PyCharm
import json

from paddleocr import PaddleOCR, TextRecognition
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from tea_recognition.core.text_processor import TeaTextProcessor

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR引擎封装"""
    def __init__(self, yaml_file_path: str ):
        """
        初始化PaddleOCR
        :param use_angle_cls: 是否使用方向分类器
        :param lang: 识别语言，'ch'中文，'en'英文
        :param use_gpu: 是否使用GPU
        """
        self.ocr = PaddleOCR(paddlex_config=yaml_file_path)
        self.text_processor = TeaTextProcessor()
        logger.info("PaddleOCR引擎初始化完成。")

    def recognize_from_url(self, image_url: str) -> Dict:
        """
        从URL识别图片中的文字，并结构化为字典
        模拟你之前阿里云OCR返回的JSON格式
        """
        import requests
        from io import BytesIO

        try:
            # 1. 下载图片
            response = requests.get(image_url, timeout=10)
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"无法从URL解码图片: {image_url}")
                return {}

            # 2. 执行OCR识别
            return self.ocr.predict(input=img)

            # # 可视化结果并保存 json 结果
            # for res in result:
            #     res.print()
            #     res.save_to_img("output")
            #     res.save_to_json("output")
            # result_json = result[0].json.get('res')
            # rec_texts = result_json['rec_texts']
            #
            # # 使用智能处理器替代原来的简单逻辑
            # tea_info = self.text_processor.process_ocr_texts(rec_texts)
            #
            # return {
            #     "success": tea_info["success"],
            #     "tea_name": tea_info["tea_name"],
            #     "tea_type": tea_info["tea_type"],
            #     "confidence": tea_info["confidence"],
            #     "raw_ocr_texts": tea_info["raw_ocr_texts"],
            #     "filtered_texts": tea_info["filtered_texts"],
            #     "extraction_method": tea_info["extraction_method"]
            # }
        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return {}

    def recognize_from_path(self, image_path: str) -> Union[dict[Any, Any], tuple[str, float]]:
        """从本地文件路径识别"""
        try:
            # 使用numpy读取文件以解决中文路径问题
            import numpy as np
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return {}

            # 对示例图像执行 OCR 推理
            return self.ocr.predict(input=img)



        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return {}



# 使用示例
if __name__ == "__main__":
    # 测试本地图片
    ocr_engine = PaddleOCREngine(yaml_file_path="PaddleOCR.yaml")
    result = ocr_engine.recognize_from_url("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")
    print("OCR结果:", result)

