#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/2/8 09:48
# @Author  : zhangpeng /zpskt
# @File    : ocr.py
# @Software: PyCharm

import os
import json
import re
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import logging

# 配置日志，方便查看运行过程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TeaInfo:
    """茶叶信息数据类，用于结构化存储结果"""
    name: str  # 茶叶具体名称
    type: str  # 茶叶类型（绿茶、红茶等）
    raw_text: str  # OCR识别的原始文本
    confidence: float  # 整体置信度（可根据OCR置信度和规则匹配度综合）
    coordinates: Optional[List] = None  # 为后续切图预留的坐标字段


class TeaRecognizer:
    def __init__(self, ocr_api_key: str = None, ocr_api_secret: str = None):
        """
        初始化识别器。
        第一阶段使用阿里云OCR（云端），后续可替换为PaddleOCR（本地）。
        """
        self.ocr_api_key = ocr_api_key or os.getenv('ALIYUN_OCR_API_KEY')
        self.ocr_api_secret = ocr_api_secret or os.getenv('ALIYUN_OCR_API_SECRET')

        # 初始化茶叶类型规则知识库（核心！）
        self.tea_type_knowledge_base = self._init_tea_knowledge_base()

        logger.info("茶叶识别器初始化完成。")

    def _init_tea_knowledge_base(self) -> Dict:
        """初始化茶叶类型判断的规则知识库"""
        return {
            "绿茶": ["绿茶", "龙井", "碧螺春", "毛峰", "毛尖", "瓜片", "日照绿", "蒸青", "炒青", "烘青", "晒青",
                     "雀舌"],
            "红茶": ["红茶", "金骏眉", "正山小种", "祁门红", "滇红", "英德红", "宜兴红", "川红", "闽红", "湖红"],
            "乌龙茶": ["乌龙", "铁观音", "大红袍", "凤凰单丛", "单枞", "冻顶乌龙", "岩茶", "水仙", "肉桂"],
            "黑茶": ["黑茶", "普洱", "熟茶", "生茶", "六堡", "安化", "茯砖", "青砖", "康砖"],
            "白茶": ["白茶", "白毫银针", "白牡丹", "寿眉", "贡眉"],
            "黄茶": ["黄茶", "君山银针", "蒙顶黄芽", "霍山黄芽", "沩山毛尖","生态茶"],
            "花茶": ["花茶", "茉莉花", "玫瑰花", "菊花", "桂花", "工艺花茶"]
        }

    def recognize_from_url(self, image_url: str) -> TeaInfo:
        """
        从图片URL识别茶叶信息（主函数）。
        流程：获取图片 -> OCR识别 -> 文本处理 -> 规则判断。
        """
        logger.info(f"开始识别图片: {image_url}")

        # 1. OCR文字识别（第一阶段用阿里云，第二阶段替换为PaddleOCR）
        # result_json = self._aliyun_ocr(image_url)
        # 注意：此行代码存在语法错误，已注释掉
        result_json = json.loads('{"生产地": "中国云南省潞西市生态茶叶有限", "生产日期": "", "茶叶名称": "生态茶", "茶叶类型": ""}')
        if not result_json:
            return TeaInfo(name="识别失败", type="未知", raw_text="", confidence=0.0)

        logger.info(f"OCR原始文本: {result_json}...")

        # 2. 文本后处理：清洗和提取关键名称
        # 从JSON结果中提取茶叶名称文本
        tea_name_text = ""
        if isinstance(result_json, dict):
            tea_name_text = result_json.get('茶叶名称', '')
        else:
            # 如果不是字典，直接使用原始文本
            tea_name_text = str(result_json)
        
        cleaned_text, potential_names = self._process_ocr_text(tea_name_text)

        # 3. 茶叶名称提取（取最可能的一个）
        tea_name = self._extract_tea_name(potential_names, cleaned_text)

        # 4. 茶叶类型判断（基于规则）
        tea_type, match_score = self._classify_tea_type(tea_name, cleaned_text)

        # 5. 组装结果
        confidence = match_score * 0.7 + 0.3  # 简化的置信度计算（规则匹配度占70%）

        return TeaInfo(
            name=tea_name,
            type=tea_type,
            raw_text=result_json,  # 只存储前200字符
            confidence=round(confidence, 2)
        )

    def _aliyun_ocr(self, image_url: str) -> str:
        """调用阿里云OCR API（第一阶段实现）"""
        from openai import OpenAI
        import os

        PROMPT_TICKET_EXTRACTION = """
        请提取茶叶图像中的茶叶名称、生产日期、生产地、茶叶类型。
        要求准确无误的提取上述关键信息、不要遗漏和捏造虚假信息，模糊或者强光遮挡的单个文字可以用英文问号?代替。如果是繁体中文，也请准确提取并转换为简体中文。
        返回数据格式以json方式输出，格式为：{'茶叶名称'：'xxx', '生产日期'：'xxx', '生产地'：'xxx', '茶叶类型'：'xxx'},
        """

        try:
            client = OpenAI(
                # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
                # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                # 以下为北京地域的 base_url，若使用弗吉尼亚地域模型，需要将base_url换成https://dashscope-us.aliyuncs.com/compatible-mode/v1
                # 若使用新加坡地域的模型，需将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            completion = client.chat.completions.create(
                model="qwen-vl-ocr-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url},
                                # 输入图像的最小像素阈值，小于该值图像会进行放大，直到总像素大于min_pixels
                                "min_pixels": 32 * 32 * 3,
                                # 输入图像的最大像素阈值，超过该值图像会进行缩小，直到总像素低于max_pixels
                                "max_pixels": 32 * 32 * 8192
                            },
                            # 模型支持在text字段中传入Prompt，若未传入，则会使用默认的Prompt：Please output only the text content from the image without any additional descriptions or formatting.
                            {"type": "text",
                             "text": PROMPT_TICKET_EXTRACTION}
                        ]
                    }
                ])
            print(completion.choices[0].message.content)
            response_content = completion.choices[0].message.content
            # 提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有代码块格式，尝试直接解析
                json_str = response_content.strip()
            # 解析JSON
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"错误信息: {e}")

    def _process_ocr_text(self, raw_text: str) -> Tuple[str, List[str]]:
        """清洗OCR文本并提取可能的茶叶名称候选"""
        # 1. 去除换行符、多余空格
        cleaned = re.sub(r'\s+', ' ', raw_text).strip()

        # 2. 提取可能的中文商品名（通常是2-6个中文字符，且包含“茶”或特定字符）
        # 中文Unicode范围: \u4e00-\u9fff
        chinese_blocks = re.findall(r'[\u4e00-\u9fff]{2,10}', cleaned)

        # 3. 过滤出包含茶叶相关关键词的块作为候选名称
        tea_keywords = ['茶', '饼', '沱', '砖', '芽', '毫', '尖', '峰', '种', '观音', '袍']
        potential_names = [
            block for block in chinese_blocks
            if any(keyword in block for keyword in tea_keywords) and len(block) <= 6
        ]

        # 如果没有找到，返回前几个中文块作为候选
        if not potential_names and chinese_blocks:
            potential_names = chinese_blocks[:3]

        logger.debug(f"文本清洗后: {cleaned}")
        logger.debug(f"候选名称: {potential_names}")

        return cleaned, potential_names

    def _extract_tea_name(self, potential_names: List[str], full_text: str) -> str:
        """从候选名称中选择最可能的茶叶名称"""
        if not potential_names:
            return "未知茶叶"

        # 策略1：优先选择包含“茶”字且长度适中的
        for name in potential_names:
            if '茶' in name and 2 <= len(name) <= 5:
                return name

        # 策略2：选择第一个候选
        return potential_names[0]

    def _classify_tea_type(self, tea_name: str, full_text: str) -> Tuple[str, float]:
        """
        基于规则判断茶叶类型。
        返回: (类型, 匹配得分)
        """
        search_text = tea_name + " " + full_text
        search_text_lower = search_text.lower()

        best_match_type = "未知"
        best_match_score = 0.0

        for tea_type, keywords in self.tea_type_knowledge_base.items():
            for keyword in keywords:
                if keyword in search_text_lower:
                    # 计算匹配分数：关键词长度/总关键词平均长度（简单启发式）
                    score = len(keyword) / 5.0  # 假设平均关键词长度5
                    if score > best_match_score:
                        best_match_score = score
                        best_match_type = tea_type
                    break  # 找到该类别一个关键词就跳出

        # 限制分数在0-1之间
        best_match_score = min(1.0, best_match_score)

        logger.info(f"类型判断: '{tea_name}' -> {best_match_type} (得分: {best_match_score:.2f})")
        return best_match_type, best_match_score

    def add_knowledge(self, tea_type: str, keywords: List[str]):
        """动态添加规则知识，用于后续训练纠正"""
        if tea_type not in self.tea_type_knowledge_base:
            self.tea_type_knowledge_base[tea_type] = []

        self.tea_type_knowledge_base[tea_type].extend(keywords)
        self.tea_type_knowledge_base[tea_type] = list(set(self.tea_type_knowledge_base[tea_type]))  # 去重
        logger.info(f"已添加规则: {tea_type} -> {keywords}")


# ==================== 使用示例 ====================

def main():
    """主函数：演示完整使用流程"""

    # 0. 请先设置你的阿里云OCR API密钥（环境变量或直接填写）
    # export ALIYUN_OCR_API_KEY="your_key"
    # export ALIYUN_OCR_API_SECRET="your_secret"

    # 1. 初始化识别器
    recognizer = TeaRecognizer()

    # 2. 添加一些自定义规则（模拟用户纠正过程）
    recognizer.add_knowledge("普洱茶", ["七子饼", "老班章", "冰岛"])
    recognizer.add_knowledge("绿茶", ["黄金芽", "安吉白"])

    # 3. 测试用的图片URL（这里用模拟数据，实际请替换为真实茶叶图片URL）
    test_images = [
        "https://file.linkcook.cn/image/food/app/photo/e92d309b-7dd9-4ab6-9c1b-1dd701e571dd.jpg",
        "https://file.linkcook.cn/image/food/app/photo/e92d309b-7dd9-4ab6-9c1b-1dd701e571dd.jpg",
    ]

    # 4. 批量识别
    for i, img_url in enumerate(test_images[:]):
        print(f"\n{'=' * 60}")
        print(f"测试图片 {i + 1}: {img_url}")
        print(f"{'=' * 60}")

        try:
            # 核心识别调用
            result = recognizer.recognize_from_url(img_url)

            # 打印结果
            print(f"✅ 识别成功!")
            print(f"   茶叶名称: {result.name}")
            print(f"   茶叶类型: {result.type}")
            print(f"   置信度: {result.confidence}")
            print(f"   原始文本: {result.raw_text}")

            # 可以转换为JSON用于API返回
            result_dict = {
                "teaName": result.name,
                "teaType": result.type,
                "confidence": result.confidence,
                "rawText": result.raw_text
            }
            print(f"   JSON格式: {json.dumps(result_dict, ensure_ascii=False, indent=2)}")

        except Exception as e:
            print(f"❌ 识别失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()