#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : text_processor_optimized.py
# @Software: PyCharm
import re
import jieba
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter
import json
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TeaTextProcessor:
    """针对你的OCR格式优化的茶叶文本处理器"""

    def __init__(self):
        # 加载茶叶知识库
        self.tea_knowledge_base = self._load_tea_knowledge()

        # 加载噪声词库（特别针对你的OCR问题）
        self.noise_words = self._load_noise_words()

        # 加载jieba自定义词典
        self._init_jieba()

        logger.info("优化版茶叶文本处理器初始化完成")

    def _load_tea_knowledge(self) -> Dict:
        """加载茶叶知识库"""
        return {
            "绿茶": ["绿茶", "龙井", "碧螺春", "毛峰", "毛尖", "瓜片", "日照绿", "太平猴魁",
                     "六安瓜片", "信阳毛尖", "黄山毛峰", "庐山云雾", "安吉白茶", "黄金芽", "日照绿茶"],
            "红茶": ["红茶", "金骏眉", "正山小种", "祁门红茶", "滇红", "川红", "宜红",
                     "英德红茶", "九曲红梅", "政和工夫", "正山小钟"],
            "乌龙茶": ["乌龙", "铁观音", "大红袍", "凤凰单丛", "单枞", "冻顶乌龙", "武夷岩茶",
                       "水仙", "肉桂", "白鸡冠", "铁罗汉"],
            "黑茶": ["黑茶", "普洱", "熟茶", "生茶", "六堡茶", "安化黑茶", "茯砖", "青砖",
                     "千两茶", "沱茶", "饼茶", "七子饼", "生态茶", "七子饼茶"],
            "白茶": ["白茶", "白毫银针", "白牡丹", "寿眉", "贡眉", "月光白"],
            "黄茶": ["黄茶", "君山银针", "蒙顶黄芽", "霍山黄芽"]
        }

    def _load_noise_words(self) -> Set[str]:
        """加载针对你OCR的噪声词库"""
        noise = {
            # 系统通知相关（从你的OCR数据中提取）
            "Haier", "海尔", "智家", "冰箱", "温度", "异常", "提醒", "检查", "舱室", "问题",
            "蓝牙", "定位", "服务", "录屏", "自动", "亮度", "蓝牙", "定位服务",
            "16:04", "15:30", "12:00",  # 时间
            "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月",
            "周一", "周二", "周三", "周四", "周五", "周六", "周日",
            "乙巳年", "丙午年", "丁未年",  # 农历年份
            "腊月", "正月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "冬月",

            # 通用噪声
            "提醒", "通知", "消息", "警告", "错误", "异常",
            "检查", "处理", "解决", "设置", "配置",
            "...", "..", "·", "×", "...", "..",  # 省略号和特殊字符

            # 计量单位
            "g", "kg", "ml", "L", "克", "千克", "毫升", "升",

            # 包装词
            "净含量", "保质期", "生产日期", "许可证", "执行标准", "地址", "电话",

            # 通用品牌（不是茶叶的）
            "Haier-Z", "海尔智家"
        }

        # 添加所有可能的组合
        additional_noise = set()
        for word in list(noise):
            if len(word) > 1:
                # 添加可能的变体
                additional_noise.add(word.lower())
                additional_noise.add(word.upper())

        noise.update(additional_noise)
        return noise

    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加茶叶专业词汇
        tea_words = []
        for tea_type, keywords in self.tea_knowledge_base.items():
            tea_words.extend(keywords)

        for word in tea_words:
            if len(word) >= 2:  # 只添加长度>=2的词
                jieba.add_word(word, freq=1000, tag='n')

        # 添加茶叶产地
        locations = ["云南", "福建", "浙江", "安徽", "四川", "广东", "湖南", "广西",
                     "山东", "江苏", "江西", "湖北", "陕西", "贵州"]
        for loc in locations:
            jieba.add_word(loc, freq=500, tag='ns')

        # 添加茶叶包装相关词
        packaging = ["七子饼", "沱茶", "砖茶", "散茶", "礼盒", "罐装", "袋泡", "小包装"]
        for word in packaging:
            jieba.add_word(word, freq=300, tag='n')

    def process_ocr_texts(self, ocr_texts: List[str]) -> Dict:
        """
        处理你的OCR返回的纯字符串列表

        Args:
            ocr_texts: ['16:04', '1月22日周四', 'Haier-Z...', ...]

        Returns:
            完整的茶叶识别结果
        """
        logger.info(f"收到{len(ocr_texts)}个OCR文本块")
        logger.debug(f"原始OCR: {ocr_texts}")

        # 1. 文本清洗和过滤
        filtered_texts = self._filter_and_clean(ocr_texts)
        logger.info(f"过滤后保留{len(filtered_texts)}个文本块")

        # 2. 提取茶叶名称
        tea_name_result = self._extract_tea_name_from_list(filtered_texts)

        # 3. 确定茶叶类型
        tea_type, type_conf = self._determine_tea_type(
            tea_name_result["name"],
            tea_name_result.get("context", "")
        )

        # 4. 组装结果
        return {
            "success": True,
            "tea_name": tea_name_result["name"],
            "tea_type": tea_type,
            "confidence": tea_name_result.get("confidence", 0.5),
            "type_confidence": type_conf,
            "extraction_method": tea_name_result.get("method", "unknown"),
            "filtered_texts": filtered_texts,
            "raw_ocr_texts": ocr_texts,
            "candidates": tea_name_result.get("candidates", []),
            "timestamp": datetime.now().isoformat()
        }

    def _filter_and_clean(self, texts: List[str]) -> List[str]:
        """过滤和清洗文本"""
        filtered = []

        for text in texts:
            # 跳过空文本
            if not text or len(text.strip()) == 0:
                continue

            text = text.strip()

            # 跳过纯噪声文本
            if self._is_pure_noise(text):
                continue

            # 跳过时间格式
            if self._is_time_format(text):
                continue

            # 跳过系统通知片段
            if self._is_system_notification(text):
                continue

            # 跳过过短的非中文文本
            if len(text) < 2 and not self._contains_chinese(text):
                continue

            # 清理特殊字符但保留中文和基本标点
            cleaned = re.sub(r'[…\.]{2,}', '', text)  # 移除连续的...或..
            cleaned = re.sub(r'[·×※]', '', cleaned)  # 移除特殊符号

            if cleaned and len(cleaned) >= 2:
                filtered.append(cleaned)

        return filtered

    def _is_pure_noise(self, text: str) -> bool:
        """检查是否是纯噪声"""
        # 单个字符且不是中文
        if len(text) == 1 and not '\u4e00' <= text <= '\u9fff':
            return True

        # 全是噪声词
        words = jieba.lcut(text)
        noise_count = sum(1 for word in words if word in self.noise_words)
        return noise_count >= len(words) * 0.8  # 80%以上是噪声词

    def _is_time_format(self, text: str) -> bool:
        """检查是否是时间格式"""
        patterns = [
            r'^\d{1,2}:\d{2}$',  # 16:04
            r'^\d{1,2}月\d{1,2}日',  # 1月22日
            r'周[一二三四五六日]',  # 周四
            r'^\d{4}年',  # 乙巳年
            r'^\d{1,2}月$',  # 腊月
        ]
        return any(re.search(pattern, text) for pattern in patterns)

    def _is_system_notification(self, text: str) -> bool:
        """检查是否是系统通知"""
        notification_keywords = ["提醒", "通知", "警告", "异常", "检查", "设置", "蓝牙", "定位"]
        return any(keyword in text for keyword in notification_keywords)

    def _contains_chinese(self, text: str) -> bool:
        """检查是否包含中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _extract_tea_name_from_list(self, filtered_texts: List[str]) -> Dict:
        """从过滤后的文本列表中提取茶叶名称"""
        if not filtered_texts:
            return {
                "name": "未识别到茶叶信息",
                "confidence": 0.0,
                "method": "no_text",
                "candidates": []
            }

        # 合并所有文本用于分析上下文
        context = " ".join(filtered_texts)
        logger.info(f"分析上下文: {context}")

        # 策略1：直接查找茶叶关键词
        direct_matches = self._find_direct_tea_matches(filtered_texts)

        # 策略2：查找包含"茶"字的文本
        tea_char_matches = self._find_tea_char_matches(filtered_texts)

        # 策略3：基于模式的匹配
        pattern_matches = self._find_pattern_matches(context)

        # 合并所有候选
        all_candidates = direct_matches + tea_char_matches + pattern_matches

        # 去重
        unique_candidates = []
        seen = set()
        for cand in all_candidates:
            if cand["name"] not in seen:
                unique_candidates.append(cand)
                seen.add(cand["name"])

        logger.info(f"找到{len(unique_candidates)}个候选: {[c['name'] for c in unique_candidates]}")

        # 评分和选择
        if unique_candidates:
            best = self._score_candidates(unique_candidates, context)
            return best
        else:
            # 如果没有找到茶叶相关文本，尝试其他策略
            return self._fallback_strategy(filtered_texts, context)

    def _find_direct_tea_matches(self, texts: List[str]) -> List[Dict]:
        """直接查找茶叶关键词"""
        candidates = []

        # 收集所有茶叶关键词
        all_tea_keywords = []
        for keywords in self.tea_knowledge_base.values():
            all_tea_keywords.extend(keywords)

        for text in texts:
            for keyword in all_tea_keywords:
                if keyword in text:
                    # 提取包含关键词的片段
                    start = text.find(keyword)
                    # 取关键词前后各2个字符
                    start_idx = max(0, start - 2)
                    end_idx = min(len(text), start + len(keyword) + 2)
                    extracted = text[start_idx:end_idx].strip()

                    candidates.append({
                        "name": extracted,
                        "method": "direct_keyword",
                        "keyword": keyword,
                        "score": 0.8,
                        "source": text
                    })

        return candidates

    def _find_tea_char_matches(self, texts: List[str]) -> List[Dict]:
        """查找包含'茶'字的文本"""
        candidates = []

        for text in texts:
            if '茶' in text:
                # 提取包含'茶'的合理长度片段
                tea_pos = text.find('茶')
                # 尝试取茶字前后各3个字符
                start = max(0, tea_pos - 3)
                end = min(len(text), tea_pos + 4)  # 茶字本身占1个位置
                extracted = text[start:end].strip()

                if 2 <= len(extracted) <= 8:
                    candidates.append({
                        "name": extracted,
                        "method": "tea_char",
                        "score": 0.7,
                        "source": text
                    })

        return candidates

    def _find_pattern_matches(self, context: str) -> List[Dict]:
        """基于模式匹配"""
        candidates = []

        patterns = [
            (r'[\u4e00-\u9fff]{2,6}茶', 0.7),  # XX茶
            (r'[\u4e00-\u9fff]{2,6}毛[峰尖]', 0.6),  # XX毛峰/毛尖
            (r'[\u4e00-\u9fff]{2,6}普洱', 0.8),  # XX普洱
            (r'[\u4e00-\u9fff]{2,6}观音', 0.7),  # XX观音
            (r'[\u4e00-\u9fff]{2,6}龙井', 0.8),  # XX龙井
            (r'[\u4e00-\u9fff]{2,8}饼茶', 0.6),  # XX饼茶
        ]

        for pattern, base_score in patterns:
            matches = re.findall(pattern, context)
            for match in matches:
                candidates.append({
                    "name": match,
                    "method": "pattern",
                    "score": base_score,
                    "pattern": pattern
                })

        return candidates

    def _score_candidates(self, candidates: List[Dict], context: str) -> Dict:
        """对候选进行评分"""
        scored = []

        for cand in candidates:
            score = cand["score"]
            name = cand["name"]

            # 加分项
            # 1. 名称长度适中
            if 3 <= len(name) <= 6:
                score += 0.1

            # 2. 包含明确的茶叶类型词
            for tea_type, keywords in self.tea_knowledge_base.items():
                for keyword in keywords:
                    if keyword in name:
                        score += 0.15
                        break

            # 3. 出现在上下文开头（通常是商品名位置）
            if context.startswith(name):
                score += 0.1

            # 4. 包含产地信息
            locations = ["云南", "福建", "浙江", "安徽", "山东"]
            if any(loc in name for loc in locations):
                score += 0.05

            # 减分项
            # 1. 包含数字
            if re.search(r'\d', name):
                score -= 0.1

            # 2. 包含噪声词片段
            for noise in self.noise_words:
                if noise in name and len(noise) >= 2:
                    score -= 0.05

            # 最终分数限制在0.1-1.0
            final_score = max(0.1, min(1.0, score))

            scored.append({
                **cand,
                "final_score": round(final_score, 3)
            })

        # 按分数排序
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        # 选择最佳候选
        best = scored[0] if scored else {
            "name": "未知茶叶",
            "final_score": 0.1,
            "method": "default"
        }

        # 修正常见错误
        corrected_name = self._correct_common_errors(best["name"])

        return {
            "name": corrected_name,
            "confidence": best["final_score"],
            "method": best["method"],
            "candidates": [{"name": c["name"], "score": c["final_score"]} for c in scored[:5]]
        }

    def _fallback_strategy(self, texts: List[str], context: str) -> Dict:
        """后备策略：当没有明显茶叶文本时"""
        # 策略1：找最长的中文文本
        chinese_texts = [t for t in texts if self._contains_chinese(t)]
        if chinese_texts:
            longest = max(chinese_texts, key=len)
            if len(longest) >= 4:
                return {
                    "name": longest[:8],  # 截取前8个字
                    "confidence": 0.3,
                    "method": "longest_chinese",
                    "candidates": []
                }

        # 策略2：找第一个非噪声文本
        for text in texts:
            if not self._is_pure_noise(text) and len(text) >= 2:
                return {
                    "name": text[:6],
                    "confidence": 0.2,
                    "method": "first_non_noise",
                    "candidates": []
                }

        return {
            "name": "未识别到有效文本",
            "confidence": 0.0,
            "method": "fallback_failed",
            "candidates": []
        }

    def _correct_common_errors(self, name: str) -> str:
        """修正常见OCR错误"""
        corrections = {
            "铁观因": "铁观音",
            "碧罗春": "碧螺春",
            "大紅袍": "大红袍",
            "金俊眉": "金骏眉",
            "正山小钟": "正山小种",
            "鳳凰單叢": "凤凰单丛",
            "七子并": "七子饼",
            "生态莱": "生态茶",
        }

        for wrong, right in corrections.items():
            if wrong in name:
                return name.replace(wrong, right)

        # 检查是否需要添加"茶"字
        tea_keywords_without_char = ["龙井", "碧螺春", "毛峰", "毛尖", "普洱", "铁观音",
                                     "大红袍", "金骏眉", "正山小种", "七子饼", "生态"]
        for keyword in tea_keywords_without_char:
            if keyword in name and '茶' not in name:
                # 常见搭配中不加"茶"的
                if not any(x in name for x in ["饼", "沱", "砖", "散"]):
                    return f"{name}茶"

        return name

    def _determine_tea_type(self, tea_name: str, context: str = "") -> Tuple[str, float]:
        """确定茶叶类型"""
        search_text = f"{tea_name} {context}".lower()

        best_type = "未知"
        best_score = 0.0

        for tea_type, keywords in self.tea_knowledge_base.items():
            for keyword in keywords:
                if keyword.lower() in search_text:
                    # 计算匹配分数
                    score = len(keyword) / 6.0  # 关键词越长，分数越高

                    # 如果在茶叶名称中直接匹配，加分
                    if keyword.lower() in tea_name.lower():
                        score += 0.2

                    if score > best_score:
                        best_score = min(score, 1.0)
                        best_type = tea_type

                    break  # 每个类型只用一个关键词

        return best_type, round(best_score, 2)


# ==================== 测试函数 ====================

def test_with_your_data():
    """用你的数据测试"""
    print("=" * 60)
    print("测试你的OCR数据")
    print("=" * 60)

    processor = TeaTextProcessor()

    # 你的OCR数据
    your_ocr_data = [
        '16:04', '1月22日周四', '乙巳年腊月初四',  '自动亮度', '16:04',
        '·', '×'
    ]

    # 模拟一些可能的茶叶文本（根据你的实际图片）
    # 这里添加几个假设的茶叶相关文本，你需要替换为实际OCR识别出的
    test_data_with_tea = your_ocr_data + [
        '云南七子饼茶',  # 假设这是茶叶名称
        '普洱茶',  # 茶叶类型
        '357克',  # 净含量
        '云南昆明'  # 产地
    ]

    print(f"测试数据（{len(test_data_with_tea)}项）:")
    for i, text in enumerate(test_data_with_tea):
        print(f"  {i + 1:2d}. {text}")

    print("\n处理中...")
    result = processor.process_ocr_texts(test_data_with_tea)

    print("\n处理结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


def test_real_scenario():
    """真实场景测试"""
    print("\n" + "=" * 60)
    print("真实场景测试")
    print("=" * 60)

    processor = TeaTextProcessor()

    # 模拟不同的茶叶包装OCR结果
    test_cases = [
        {
            "name": "普洱茶案例",
            "ocr_texts": [
                '16:30', '海尔智家', '蓝牙已连接', '云南七子饼茶', '普洱茶（生茶）',
                '净含量: 357g', '生产日期: 2023-05-10', '云南昆明', '...'
            ]
        },
        {
            "name": "绿茶案例",
            "ocr_texts": [
                '09:15', '录屏中', '日照绿茶', '特级绿茶', '山东日照',
                '250克', '生产许可证: SC123', '...', '·'
            ]
        },
        {
            "name": "乌龙茶案例",
            "ocr_texts": [
                '14:22', '定位服务开启', '安溪铁观音', '乌龙茶', '特级',
                '福建安溪', '500克', '...', '×'
            ]
        },
        {
            "name": "噪声过多案例（你的情况）",
            "ocr_texts": [
                '16:04', '1月22日周四', 'Haier-Z...', '蓝牙', '定位服务',
                '录屏', '自动亮度', '海尔智家', 'Haier', '冰箱温度异常提醒',
                '请检查您家的冰箱各舱室温度是否正常', '·', '×'
            ]
        }
    ]

    for case in test_cases:
        print(f"\n测试: {case['name']}")
        result = processor.process_ocr_texts(case['ocr_texts'])
        print(f"  茶叶名称: {result['tea_name']} (置信度: {result['confidence']:.2f})")
        print(f"  茶叶类型: {result['tea_type']} (置信度: {result['type_confidence']:.2f})")
        print(f"  提取方法: {result['extraction_method']}")


def integrate_with_your_ocr():
    """如何集成到你的OCR引擎"""
    print("\n" + "=" * 60)
    print("集成指南")
    print("=" * 60)

    integration_code = '''
# 在你的 ocr_engine.py 中

from text_processor_optimized import TeaTextProcessor

class PaddleOCREngine:
    def __init__(self):
        # ... 原有的OCR初始化 ...
        self.ocr = PaddleOCR(...)
        # 添加文本处理器
        self.text_processor = TeaTextProcessor()

    def recognize_from_url(self, image_url: str) -> Dict:
        try:
            # ... 原有的图片下载和OCR识别 ...
            # 假设你的OCR返回 result['rec_texts'] 是字符串列表

            ocr_texts = result['rec_texts']  # 这是你的字符串列表

            # 使用优化的文本处理器
            tea_info = self.text_processor.process_ocr_texts(ocr_texts)

            return {
                "success": tea_info["success"],
                "tea_name": tea_info["tea_name"],
                "tea_type": tea_info["tea_type"],
                "confidence": tea_info["confidence"],
                "raw_ocr_texts": tea_info["raw_ocr_texts"],
                "filtered_texts": tea_info["filtered_texts"],
                "extraction_method": tea_info["extraction_method"]
            }

        except Exception as e:
            return {"error": str(e), "success": False}
    '''

    print(integration_code)
    print("\n只需要将 TeaTextProcessor 集成到你的现有OCR引擎中即可！")


if __name__ == "__main__":
    # 测试你的数据
    test_with_your_data()

    # 测试更多场景
    test_real_scenario()

    # 显示集成方法
    integrate_with_your_ocr()

    print("\n" + "=" * 60)
    print("✅ 优化完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 运行此脚本测试效果")
    print("2. 将 TeaTextProcessor 集成到你的 ocr_engine.py")
    print("3. 用真实茶叶图片测试")
    print("4. 根据测试结果调整 noise_words 和 tea_knowledge_base")
