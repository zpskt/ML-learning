"""
方案架构：
OCR模型（识别文字） → 茶叶名称 → 茶叶分类模型 → 茶叶类型
                     ↓
                知识库/LLM增强 ← 如有错误，记录并训练
"""

import json
import requests
from typing import Dict, List, Tuple
import hashlib
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TeaInfo:
    """茶叶信息结构"""
    name: str  # 茶叶名称
    type: str  # 茶叶类型
    confidence: float  # 置信度
    source_image: str  # 源图片标识


class AliVisionTeaRecognizer:
    def __init__(self, access_key_id: str, access_key_secret: str):
        """
        初始化阿里云视觉服务
        文档：https://help.aliyun.com/zh/model-studio/vision
        """
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = "https://visionai.cn-hangzhou.aliyuncs.com"

        # 模型服务端点
        self.ocr_endpoint = "ocr-api.cn-hangzhou.aliyuncs.com"
        self.classify_endpoint = "imageenhan.cn-hangzhou.aliyuncs.com"

        # 错误案例收集
        self.error_cases = []
        self.correction_db = self._load_correction_db()

    def recognize_tea_complete(self, image_url: str) -> Dict:
        """
        完整识别流程：
        1. OCR识别茶叶包装文字
        2. 提取茶叶名称
        3. 判断茶叶类型（使用分类模型+规则）
        4. 返回结构化结果
        """
        # 1. OCR识别
        ocr_text = self._call_aliyun_ocr(image_url)

        # 2. 提取茶叶名称（从OCR结果中）
        tea_name = self._extract_tea_name(ocr_text)

        # 3. 判断茶叶类型（多种方法结合）
        tea_type, confidence = self._classify_tea_type(image_url, tea_name, ocr_text)

        # 4. 检查是否需要修正
        corrected_result = self._check_and_correct(tea_name, tea_type, image_url)
        if corrected_result:
            tea_name, tea_type = corrected_result

        return {
            "茶叶名称": tea_name,
            "茶叶类型": tea_type,
            "置信度": confidence,
            "OCR文本": ocr_text,
            "识别时间": datetime.now().isoformat()
        }

    def _call_aliyun_ocr(self, image_url: str) -> str:
        """调用阿里云OCR API"""
        # 这里使用阿里云视觉OCR服务
        # 实际调用需要安装SDK: pip install alibabacloud_ocr_api20210707
        try:
            # 示例代码 - 实际需要按照阿里云文档配置
            import alibabacloud_ocr_api20210707 as ocr
            from alibabacloud_tea_openapi import models as open_models

            config = open_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.ocr_endpoint
            )

            client = ocr.Client(config)
            request = ocr.models.RecognizeAdvancedRequest(
                url=image_url
            )

            response = client.recognize_advanced(request)
            result = json.loads(response.body)

            # 提取所有文本
            all_text = ""
            if 'data' in result and 'prism_wordsInfo' in result['data']:
                for word_info in result['data']['prism_wordsInfo']:
                    all_text += word_info.get('word', '') + " "

            return all_text.strip()

        except Exception as e:
            print(f"OCR识别失败: {e}")
            # 可以降级到其他OCR服务
            return self._fallback_ocr(image_url)

    def _extract_tea_name(self, ocr_text: str) -> str:
        """从OCR文本中提取茶叶名称"""
        # 方法1：查找包含"茶"的关键词
        keywords = ["茶", "茗", "叶", "毛", "毫", "针", "芽", "峰"]

        # 按行分割，找最可能是名称的行
        lines = ocr_text.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in keywords) and 2 <= len(line) <= 8:
                return line

        # 方法2：使用启发式规则
        # 通常是较短的中文名称（2-6字）
        possible_names = []
        for line in lines:
            line = line.strip()
            if 2 <= len(line) <= 6 and self._is_chinese_tea_name(line):
                possible_names.append(line)

        return possible_names[0] if possible_names else ocr_text[:20]

    def _classify_tea_type(self, image_url: str, tea_name: str, ocr_text: str) -> Tuple[str, float]:
        """
        判断茶叶类型：多种方法结合

        方法优先级：
        1. 使用阿里云图像分类API
        2. 使用OCR文本分析
        3. 使用本地知识库
        """

        # 方法1：使用阿里云图像分类/商品识别
        try:
            image_type = self._call_aliyun_image_classify(image_url)
            if image_type and image_type != "unknown":
                return image_type, 0.85
        except:
            pass

        # 方法2：OCR文本分析（关键词匹配）
        tea_type_by_text = self._classify_by_text(ocr_text, tea_name)
        if tea_type_by_text:
            return tea_type_by_text, 0.75

        # 方法3：使用茶叶名称的常见模式
        tea_type_by_pattern = self._classify_by_pattern(tea_name)
        if tea_type_by_pattern:
            return tea_type_by_pattern, 0.65

        return "未知类型", 0.5

    def _call_aliyun_image_classify(self, image_url: str) -> str:
        """调用阿里云图像分类API"""
        # 阿里云商品识别或图像标签服务
        # 实际调用需要相应SDK
        try:
            # 示例：使用图像标签服务
            # pip install alibabacloud_imagerecog20190930
            import alibabacloud_imagerecog20190930 as imagerecog
            from alibabacloud_tea_openapi import models as open_models

            config = open_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.classify_endpoint
            )

            client = imagerecog.Client(config)
            request = imagerecog.models.TaggingImageRequest(
                image_url=image_url
            )

            response = client.tagging_image(request)
            result = json.loads(response.body)

            # 解析标签，找茶叶相关标签
            tags = result.get('data', {}).get('tags', [])
            for tag in tags:
                tag_name = tag.get('value', '')
                confidence = tag.get('confidence', 0)

                if confidence > 0.5:
                    # 检查是否是茶叶类型
                    tea_type = self._map_tag_to_tea_type(tag_name)
                    if tea_type:
                        return tea_type

        except Exception as e:
            print(f"图像分类失败: {e}")

        return ""

    def _classify_by_text(self, ocr_text: str, tea_name: str) -> str:
        """通过文本分析判断茶叶类型"""
        text = ocr_text.lower() + " " + tea_name.lower()

        # 六大茶类的关键词
        tea_type_keywords = {
            "绿茶": ["绿", "蒸青", "炒青", "烘青", "晒青", "龙井", "碧螺春", "毛峰"],
            "红茶": ["红", "全发酵", "祁红", "滇红", "正山小种", "金骏眉", "英德红"],
            "乌龙茶": ["乌龙", "青茶", "半发酵", "铁观音", "大红袍", "凤凰单枞", "冻顶"],
            "黑茶": ["黑", "后发酵", "普洱", "熟茶", "安化", "六堡", "茯砖"],
            "白茶": ["白", "微发酵", "白毫银针", "白牡丹", "寿眉", "贡眉"],
            "黄茶": ["黄", "闷黄", "君山银针", "蒙顶黄芽", "霍山黄芽"]
        }

        for tea_type, keywords in tea_type_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return tea_type

        return ""

    def _classify_by_pattern(self, tea_name: str) -> str:
        """根据茶叶名称模式判断"""
        # 常见茶叶名称模式
        patterns = {
            "绿茶": ["绿茶", "毛尖", "毛峰", "龙井", "碧螺春", "瓜片"],
            "红茶": ["红茶", "红袍", "红芽", "红韵"],
            "普洱茶": ["普洱", "老班章", "冰岛", "易武"],
            "白茶": ["白茶", "白毫", "白牡丹"],
            "乌龙茶": ["观音", "岩茶", "单枞", "乌龙"]
        }

        for tea_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in tea_name:
                    return tea_type

        return ""

    def _check_and_correct(self, tea_name: str, tea_type: str, image_url: str) -> Tuple[str, str]:
        """检查并修正识别结果"""
        image_hash = self._get_image_hash(image_url)

        # 检查修正数据库
        if image_hash in self.correction_db:
            correction = self.correction_db[image_hash]
            return correction["name"], correction["type"]

        return None

    def record_error_and_train(self, image_url: str, correct_name: str, correct_type: str):
        """
        记录错误并准备训练数据

        参数：
        - image_url: 图片地址
        - correct_name: 正确的茶叶名称
        - correct_type: 正确的茶叶类型
        """
        image_hash = self._get_image_hash(image_url)

        # 1. 记录错误案例
        self.error_cases.append({
            "image_url": image_url,
            "image_hash": image_hash,
            "correct_name": correct_name,
            "correct_type": correct_type,
            "timestamp": datetime.now().isoformat()
        })

        # 2. 添加到修正数据库（立即生效）
        self.correction_db[image_hash] = {
            "name": correct_name,
            "type": correct_type
        }

        # 3. 保存训练数据
        self._save_training_data(image_url, correct_name, correct_type)

        print(f"已记录错误案例，积累 {len(self.error_cases)} 个待训练样本")

        # 4. 如果积累足够样本，触发重新训练
        if len(self.error_cases) >= 10:  # 设置阈值
            self._trigger_retraining()

    def _save_training_data(self, image_url: str, name: str, type: str):
        """保存训练数据到文件"""
        training_data = {
            "image_url": image_url,
            "label": {
                "tea_name": name,
                "tea_type": type
            },
            "annotation_time": datetime.now().isoformat()
        }

        # 保存到JSON文件
        with open("tea_training_data.json", "a", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False)
            f.write("\n")

    def _trigger_retraining(self):
        """触发重新训练"""
        print("积累足够错误案例，开始重新训练...")

        # 1. 准备训练数据
        training_file = self._prepare_training_dataset()

        # 2. 调用阿里云模型训练API
        # 实际需要按照阿里云自定义模型训练流程
        self._train_aliyun_model(training_file)

        # 3. 清空错误案例
        self.error_cases = []

    def _prepare_training_dataset(self) -> str:
        """准备训练数据集"""
        # 将错误案例转换为训练格式
        training_samples = []

        for case in self.error_cases:
            sample = {
                "image_url": case["image_url"],
                "annotation": {
                    "tea_name": case["correct_name"],
                    "tea_type": case["correct_type"]
                }
            }
            training_samples.append(sample)

        # 保存为训练集文件
        output_file = f"tea_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)

        return output_file

    def _train_aliyun_model(self, training_file: str):
        """调用阿里云模型训练服务"""
        # 阿里云视觉理解平台支持自定义模型训练
        # 需要按照官方文档操作

        print(f"开始训练模型，使用数据文件: {training_file}")

        # 实际步骤：
        # 1. 上传训练数据到OSS
        # 2. 创建训练任务
        # 3. 监控训练进度
        # 4. 部署新模型

        # 示例代码结构：
        """
        from alibabacloud_viapi20230117.client import Client as viapiClient
        from alibabacloud_tea_openapi import models as open_models

        # 创建训练任务
        config = open_models.Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            endpoint='viapi.cn-hangzhou.aliyuncs.com'
        )

        client = viapiClient(config)
        # 调用相应的训练API
        """

        print("训练任务已提交，请到阿里云控制台查看进度")

    def _get_image_hash(self, image_url: str) -> str:
        """生成图片唯一哈希"""
        # 可以使用URL的MD5作为标识
        return hashlib.md5(image_url.encode()).hexdigest()[:16]

    def _load_correction_db(self) -> Dict:
        """加载修正数据库"""
        try:
            with open("correction_db.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def save_correction_db(self):
        """保存修正数据库"""
        with open("correction_db.json", "w", encoding="utf-8") as f:
            json.dump(self.correction_db, f, ensure_ascii=False, indent=2)

    def _is_chinese_tea_name(self, text: str) -> bool:
        """判断是否是中文茶叶名称"""
        # 简单的启发式规则
        if not text:
            return False

        # 茶叶名称通常不包含特定符号
        invalid_chars = ["：", ":", "、", "，", ",", "。", ".", "·", "@", "#", "$", "%"]
        if any(char in text for char in invalid_chars):
            return False

        # 长度通常在2-6个汉字
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return 2 <= chinese_count <= 6

    def _map_tag_to_tea_type(self, tag: str) -> str:
        """将阿里云标签映射到茶叶类型"""
        mapping = {
            "绿茶": ["绿茶", "green tea", "龙井茶", "碧螺春"],
            "红茶": ["红茶", "black tea", "祁门红茶"],
            "普洱茶": ["普洱茶", "puer tea", "普洱"],
            "白茶": ["白茶", "white tea"],
            "乌龙茶": ["乌龙茶", "oolong tea", "铁观音"],
            "花茶": ["花茶", "flower tea", "茉莉花茶"]
        }

        for tea_type, keywords in mapping.items():
            if any(keyword in tag.lower() for keyword in keywords):
                return tea_type

        return ""

    def _fallback_ocr(self, image_url: str) -> str:
        """备用OCR方案"""
        try:
            # 可以使用其他OCR服务，如百度OCR、腾讯OCR等
            # 或者本地OCR库如PaddleOCR
            import paddleocr
            ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch')
            result = ocr.ocr(image_url, cls=True)

            text = ""
            for line in result:
                if line and line[1]:
                    text += line[1][0] + " "

            return text.strip()
        except:
            return ""


# 使用示例
def main():
    # 初始化识别器
    recognizer = AliVisionTeaRecognizer(
        access_key_id="your_access_key_id",
        access_key_secret="your_access_key_secret"
    )

    # 1. 识别茶叶
    image_url = "https://example.com/tea_image.jpg"
    result = recognizer.recognize_tea_complete(image_url)

    print("识别结果:")
    print(f"茶叶名称: {result['茶叶名称']}")
    print(f"茶叶类型: {result['茶叶类型']}")
    print(f"置信度: {result['置信度']}")

    # 2. 如果识别错误，记录并训练
    # 假设识别结果错误，正确应该是"日照绿茶"和"绿茶"
    is_correct = False  # 根据实际情况判断

    if not is_correct:
        correct_name = "日照绿茶"
        correct_type = "绿茶"

        # 记录错误并加入训练数据
        recognizer.record_error_and_train(
            image_url=image_url,
            correct_name=correct_name,
            correct_type=correct_type
        )

        # 保存修正数据库
        recognizer.save_correction_db()

    # 3. 批量处理时，积累足够错误案例后自动重新训练


if __name__ == "__main__":
    main()