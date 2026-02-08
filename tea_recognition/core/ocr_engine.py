#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : ocr_engine.py
# @Software: PyCharm
import json

from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """PaddleOCR引擎封装"""

    def __init__(self, use_angle_cls=True, lang='ch', use_gpu=False):
        """
        初始化PaddleOCR
        :param use_angle_cls: 是否使用方向分类器
        :param lang: 识别语言，'ch'中文，'en'英文
        :param use_gpu: 是否使用GPU
        """
        logger.info(f"初始化PaddleOCR引擎 (语言: {lang}, GPU: {use_gpu})...")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)
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
            result = self.ocr.ocr(img, cls=True)

            # 3. 提取所有文本
            all_texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        all_texts.append((text, confidence))

            # 4. 提取关键信息（模拟结构化输出）
            structured_result = self._extract_tea_info(all_texts)

            logger.info(f"PaddleOCR识别成功，共识别{len(all_texts)}个文本块")
            return structured_result

        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return {}

    def recognize_from_path(self, image_path: str) -> Dict:
        """从本地文件路径识别"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"无法读取图片: {image_path}")
                return {}

            # 对示例图像执行 OCR 推理
            result = self.ocr.predict(input=img)

            # # 可视化结果并保存 json 结果
            # for res in result:
            #     res.print()
            #     res.save_to_img("output")
            #     res.save_to_json("output")
            result_json = result[0].json.get('res')

            structured_result = self._extract_tea_info(result_json.get('rec_texts'))
            return structured_result

        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return {}

    def _extract_tea_info(self, text_blocks: List[Tuple[str, float]]) -> Dict:
        """
        从OCR文本块中提取茶叶相关信息
        这是简化版，你可以根据实际需求增强
        """
        # 将所有文本合并
        full_text = " ".join([text for text, _ in text_blocks])

        # 简单的关键词提取逻辑（你可以替换为更复杂的NLP方法）
        result = {
            "茶叶名称": "",
            "生产日期": "",
            "生产地": "",
            "茶叶类型": "",
            "raw_text": full_text,
            "text_blocks": text_blocks
        }

        # 查找茶叶名称（包含"茶"字的文本）
        for text, confidence in text_blocks:
            if '茶' in text and len(text) <= 8:  # 假设茶叶名称较短
                result["茶叶名称"] = text
                break

        # 查找生产地（包含"省"、"市"、"产地"等关键词）
        for text, confidence in text_blocks:
            if any(keyword in text for keyword in ["省", "市", "产地", "生产", "原产地"]):
                result["生产地"] = text
                break

        # 查找生产日期（包含数字和年月日）
        import re
        date_pattern = r'\d{4}[年\.\-]\d{1,2}[月\.\-]\d{1,2}[日]?|\d{8}'
        for text, confidence in text_blocks:
            if re.search(date_pattern, text):
                result["生产日期"] = text
                break

        return result


# 使用示例
if __name__ == "__main__":
    # 测试本地图片
    ocr_engine = PaddleOCREngine()
    result = ocr_engine.recognize_from_path("/Users/zhangpeng/Desktop/zpskt/ML-learning/tea_recognition/白茶_白茶.jpg")
    print("OCR结果:", json.dumps(result, ensure_ascii=False, indent=2))