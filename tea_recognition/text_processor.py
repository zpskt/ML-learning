#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : text_processor.py
# @Software: PyCharm
import re
import jieba
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TeaTextProcessor:
    """茶叶文本智能处理器"""

    def __init__(self):
        # 加载茶叶知识库
        self.tea_knowledge_base = self._load_tea_knowledge()

        # 噪声词库（需要过滤的词）
        self.noise_words = self._load_noise_words()

        # 加载jieba自定义词典
        self._init_jieba()

        logger.info("茶叶文本处理器初始化完成")

    def _load_tea_knowledge(self) -> Dict:
        """加载茶叶知识库"""
        return {
            # 绿茶类
            "绿茶": {
                "keywords": ["绿茶", "龙井", "碧螺春", "毛峰", "毛尖", "瓜片", "日照绿", "太平猴魁",
                             "六安瓜片", "信阳毛尖", "黄山毛峰", "庐山云雾", "安吉白茶", "黄金芽"],
                "common_patterns": ["{地名}毛峰", "{地名}毛尖", "{地名}龙井"],
                "producers": ["浙江", "杭州", "苏州", "安徽", "黄山", "信阳", "日照"]
            },
            # 红茶类
            "红茶": {
                "keywords": ["红茶", "金骏眉", "正山小种", "祁门红茶", "滇红", "川红", "宜红",
                             "英德红茶", "九曲红梅", "政和工夫"],
                "common_patterns": ["祁门{级别}红茶", "云南{地名}红茶"],
                "producers": ["福建", "安徽", "云南", "四川", "广东"]
            },
            # 乌龙茶类
            "乌龙茶": {
                "keywords": ["乌龙", "铁观音", "大红袍", "凤凰单丛", "冻顶乌龙", "武夷岩茶",
                             "水仙", "肉桂", "白鸡冠", "铁罗汉"],
                "common_patterns": ["安溪铁观音", "武夷{品种}岩茶"],
                "producers": ["福建", "安溪", "武夷山", "广东", "台湾"]
            },
            # 黑茶类
            "黑茶": {
                "keywords": ["普洱", "熟茶", "生茶", "六堡茶", "安化黑茶", "茯砖", "青砖",
                             "千两茶", "沱茶", "饼茶", "七子饼"],
                "common_patterns": ["云南{年份}普洱", "{地名}黑茶", "{重量}克沱茶"],
                "producers": ["云南", "湖南", "广西", "四川"]
            },
            # 白茶类
            "白茶": {
                "keywords": ["白茶", "白毫银针", "白牡丹", "寿眉", "贡眉", "月光白"],
                "common_patterns": ["福鼎{级别}白茶", "{年份}老白茶"],
                "producers": ["福建", "福鼎", "政和"]
            }
        }

    def _load_noise_words(self) -> Set[str]:
        """加载噪声词库"""
        return {
            # 零食相关
            "零食", "饼干", "糖果", "巧克力", "薯片", "坚果", "瓜子", "花生",
            "饮料", "果汁", "牛奶", "咖啡", "奶茶", "可乐",
            # 通用包装词
            "净含量", "保质期", "生产许可证", "执行标准", "营养成分",
            "地址", "电话", "网址", "客服", "二维码",
            # 常见商标
            "康师傅", "统一", "乐事", "奥利奥", "德芙", "雀巢",
            # 计量单位
            "g", "kg", "ml", "L", "克", "千克", "毫升", "升"
        }

    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加茶叶专业词汇到jieba词典
        tea_words = []
        for tea_type, info in self.tea_knowledge_base.items():
            tea_words.extend(info["keywords"])

        for word in tea_words:
            jieba.add_word(word, freq=1000, tag='n')

        # 添加常见茶叶产地
        locations = ["云南", "福建", "浙江", "安徽", "四川", "广东", "湖南", "广西"]
        for loc in locations:
            jieba.add_word(loc, freq=500, tag='ns')

    def extract_tea_name(self, ocr_text_blocks: List[Tuple[str, float]]) -> Dict:
        """
        从OCR文本块中智能提取茶叶名称

        Args:
            ocr_text_blocks: [(文本, 置信度), ...]

        Returns:
            {
                "tea_name": "提取的茶叶名称",
                "confidence": 置信度,
                "method": "使用的提取方法",
                "candidates": ["候选名称1", "候选名称2"],
                "filtered_text": "清洗后的文本"
            }
        """
        # 1. 文本预处理
        cleaned_text = self._preprocess_text(ocr_text_blocks)
        logger.info(f"清洗后文本: {cleaned_text}")

        # 2. 多层次候选名称提取
        candidates = self._extract_candidates(cleaned_text, ocr_text_blocks)
        logger.info(f"候选名称: {candidates}")

        # 3. 候选名称评分和选择
        if candidates:
            best_candidate = self._score_and_select(candidates, cleaned_text)

            # 4. 验证和修正
            final_name = self._validate_and_correct(best_candidate["name"], cleaned_text)

            return {
                "tea_name": final_name,
                "confidence": best_candidate["score"],
                "method": best_candidate["method"],
                "candidates": [c["name"] for c in candidates],
                "filtered_text": cleaned_text,
                "raw_text_blocks": ocr_text_blocks
            }
        else:
            return {
                "tea_name": "未知茶叶",
                "confidence": 0.0,
                "method": "无候选",
                "candidates": [],
                "filtered_text": cleaned_text
            }

    def _preprocess_text(self, text_blocks: List[Tuple[str, float]]) -> str:
        """文本预处理"""
        # 1. 合并所有文本块
        all_texts = [text for text, conf in text_blocks if conf > 0.5]  # 只保留高置信度

        # 2. 去重并排序（按置信度降序）
        unique_texts = []
        seen = set()
        for text, conf in sorted(text_blocks, key=lambda x: x[1], reverse=True):
            if text not in seen:
                unique_texts.append(text)
                seen.add(text)

        # 3. 噪声过滤
        filtered_texts = []
        for text in unique_texts:
            # 检查是否包含噪声词
            if not self._contains_noise(text):
                filtered_texts.append(text)

        # 4. 合并为字符串
        combined_text = " ".join(filtered_texts)

        # 5. 清理特殊字符和空格
        cleaned = re.sub(r'\s+', ' ', combined_text).strip()
        cleaned = re.sub(r'[^\w\u4e00-\u9fff\s\-·]', '', cleaned)  # 保留中文、英文、数字、横线、间隔符

        return cleaned

    def _contains_noise(self, text: str) -> bool:
        """检查文本是否包含噪声词"""
        # 短文本直接检查
        if len(text) <= 4:
            for noise in self.noise_words:
                if noise in text:
                    return True

        # 长文本分词后检查
        words = jieba.lcut(text)
        for word in words:
            if word in self.noise_words:
                return True

        return False

    def _extract_candidates(self, cleaned_text: str,
                            original_blocks: List[Tuple[str, float]]) -> List[Dict]:
        """提取候选茶叶名称（多种策略）"""
        candidates = []

        # 策略1：直接包含"茶"字的文本块
        for text, conf in original_blocks:
            if '茶' in text and 2 <= len(text) <= 12 and not self._contains_noise(text):
                candidates.append({
                    "name": text,
                    "method": "direct_tea_char",
                    "score": conf * 0.9,
                    "source": "original_block"
                })

        # 策略2：基于茶叶关键词的匹配
        tea_keywords = []
        for tea_type, info in self.tea_knowledge_base.items():
            tea_keywords.extend(info["keywords"])

        for keyword in tea_keywords:
            if keyword in cleaned_text:
                # 提取包含关键词的短语
                pattern = rf'.{{0,6}}{keyword}.{{0,6}}'
                matches = re.findall(pattern, cleaned_text)
                for match in matches:
                    candidates.append({
                        "name": match.strip(),
                        "method": "keyword_match",
                        "score": 0.8,
                        "keyword": keyword,
                        "source": "text_scan"
                    })

        # 策略3：基于模式的匹配（如"XX茶"、"XX毛峰"）
        patterns = [
            r'[\u4e00-\u9fff]{1,4}茶',  # XX茶
            r'[\u4e00-\u9fff]{2,6}毛[峰尖]',  # XX毛峰/毛尖
            r'[\u4e00-\u9fff]{2,6}龙井',  # XX龙井
            r'[\u4e00-\u9fff]{2,6}普洱',  # XX普洱
            r'[\u4e00-\u9fff]{2,6}观音',  # XX观音（铁观音）
        ]

        for pattern in patterns:
            matches = re.findall(pattern, cleaned_text)
            for match in matches:
                candidates.append({
                    "name": match,
                    "method": "pattern_match",
                    "score": 0.7,
                    "pattern": pattern,
                    "source": "regex"
                })

        # 策略4：高频出现的中文词组
        words = jieba.lcut(cleaned_text)
        word_groups = []
        for i in range(len(words) - 1):
            if len(words[i]) >= 2 and len(words[i + 1]) >= 2:
                word_groups.append(words[i] + words[i + 1])

        if word_groups:
            word_freq = Counter(word_groups)
            common_groups = word_freq.most_common(3)
            for group, count in common_groups:
                if 4 <= len(group) <= 8:
                    candidates.append({
                        "name": group,
                        "method": "frequent_phrase",
                        "score": 0.6 * min(count / 3, 1.0),
                        "frequency": count,
                        "source": "frequency"
                    })

        # 去重
        unique_candidates = []
        seen_names = set()
        for cand in candidates:
            if cand["name"] not in seen_names:
                unique_candidates.append(cand)
                seen_names.add(cand["name"])

        return unique_candidates

    def _score_and_select(self, candidates: List[Dict], context: str) -> Dict:
        """对候选名称进行评分和选择"""
        scored_candidates = []

        for cand in candidates:
            score = cand["score"]
            name = cand["name"]

            # 加分项
            # 1. 包含明确的茶叶类型关键词
            for tea_type, info in self.tea_knowledge_base.items():
                for keyword in info["keywords"]:
                    if keyword in name:
                        score += 0.1

            # 2. 名称长度适中（3-6字最佳）
            name_len = len(name)
            if 3 <= name_len <= 6:
                score += 0.15
            elif name_len == 2 or name_len == 7:
                score += 0.05

            # 3. 出现在高频位置（通常商品名在文本前部）
            if context.startswith(name) or name in context[:50]:
                score += 0.1

            # 4. 包含产地信息（加分但不高）
            locations = ["云南", "福建", "浙江", "安徽", "四川", "广东"]
            if any(loc in name for loc in locations):
                score += 0.05

            # 减分项
            # 1. 包含数字（通常不是纯名称）
            if re.search(r'\d', name):
                score -= 0.1

            # 2. 包含英文（通常不是纯中文名称）
            if re.search(r'[a-zA-Z]', name):
                score -= 0.05

            # 限制分数范围
            score = max(0.1, min(1.0, score))

            scored_candidates.append({
                **cand,
                "final_score": round(score, 3)
            })

        # 按分数排序
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # 返回最佳候选
        return scored_candidates[0] if scored_candidates else {
            "name": "未知茶叶",
            "score": 0.0,
            "method": "default"
        }

    def _validate_and_correct(self, name: str, context: str) -> str:
        """验证和修正茶叶名称"""
        if name == "未知茶叶":
            return name

        # 1. 检查是否缺少"茶"字但明显是茶叶
        tea_keywords_no_char = ["龙井", "碧螺春", "毛峰", "毛尖", "普洱", "铁观音", "大红袍", "金骏眉"]
        for keyword in tea_keywords_no_char:
            if keyword in name and '茶' not in name:
                # 检查是否需要添加"茶"字
                if not any(kw in name for kw in ["茶", "饼", "沱", "砖"]):
                    return f"{name}茶"

        # 2. 检查是否为常见变体并标准化
        variant_mapping = {
            "铁观因": "铁观音",
            "碧罗春": "碧螺春",
            "大紅袍": "大红袍",
            "金俊眉": "金骏眉",
            "正山小鐘": "正山小种",
            "鳳凰單叢": "凤凰单丛",
        }

        for variant, standard in variant_mapping.items():
            if variant in name:
                return name.replace(variant, standard)

        return name

    def determine_tea_type(self, tea_name: str, context: str = "") -> Tuple[str, float]:
        """根据茶叶名称确定茶叶类型"""
        search_text = tea_name + " " + context
        search_text_lower = search_text.lower()

        best_match = ("未知", 0.0)

        for tea_type, info in self.tea_knowledge_base.items():
            for keyword in info["keywords"]:
                if keyword in search_text_lower:
                    # 计算匹配强度
                    score = min(1.0, len(keyword) / 5.0)

                    # 如果是精确匹配名称中的关键词，加分
                    if keyword in tea_name:
                        score += 0.3

                    if score > best_match[1]:
                        best_match = (tea_type, min(score, 1.0))
                    break  # 每个类型找到第一个匹配就跳出

        return best_match


# ==================== 使用示例 ====================

def test_processor():
    """测试处理器"""
    print("=" * 60)
    print("茶叶文本处理器测试")
    print("=" * 60)

    processor = TeaTextProcessor()

    # 测试用例
    test_cases = [
        # (模拟OCR结果, 期望的茶叶名称)
        ([
             ("康师傅绿茶", 0.95),
             ("净含量: 500ml", 0.98),
             ("生产日期: 2024-01-15", 0.97),
             ("杭州龙井茶叶有限公司", 0.90),
             ("特级龙井茶", 0.92)
         ], "特级龙井茶"),

        ([
             ("云南七子饼茶", 0.96),
             ("普洱茶 (熟茶)", 0.94),
             ("净含量: 357g", 0.98),
             ("生产许可证号: SC123", 0.85),
             ("云南省茶叶公司", 0.88)
         ], "云南七子饼茶"),

        ([
             ("乐事薯片原味", 0.97),
             ("安溪铁观音", 0.95),
             ("乌龙茶", 0.92),
             ("250克", 0.98),
             ("福建安溪", 0.89)
         ], "安溪铁观音"),

        ([
             ("饼干巧克力味", 0.96),
             ("日照绿茶", 0.94),
             ("山东特产", 0.91),
             ("2023年春茶", 0.90)
         ], "日照绿茶"),
    ]

    for i, (ocr_blocks, expected) in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"OCR输入: {ocr_blocks}")

        result = processor.extract_tea_name(ocr_blocks)

        print(f"提取结果: {result['tea_name']} (置信度: {result['confidence']:.2f})")
        print(f"提取方法: {result['method']}")
        print(f"候选列表: {result['candidates']}")

        # 确定茶叶类型
        tea_type, type_confidence = processor.determine_tea_type(
            result['tea_name'],
            result['filtered_text']
        )
        print(f"茶叶类型: {tea_type} (置信度: {type_confidence:.2f})")

        # 验证
        if result['tea_name'] == expected:
            print("✅ 匹配成功")
        else:
            print(f"⚠️  期望: {expected}, 实际: {result['tea_name']}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 测试文本处理器
    test_processor()

    print("\n\n" + "=" * 60)
    print("集成测试")
    print("=" * 60)
