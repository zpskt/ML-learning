import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from typing import Dict, Any, List, Tuple
import json
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ChartGenerator:
    """
    图表生成器类，用于将数据转换为各种类型的图表
    """
    
    def __init__(self):
        """
        初始化图表生成器
        """
        self.supported_chart_types = [
            'bar',      # 柱状图
            'line',     # 折线图
            'pie',      # 饼图
            'scatter',  # 散点图
            'histogram' # 直方图
        ]
    
    def generate_chart(self, data: str, chart_type: str, title: str = "", 
                      x_label: str = "", y_label: str = "", **kwargs) -> str:
        """
        生成图表并返回base64编码的图片
        
        Args:
            data: 查询结果数据（JSON格式字符串）
            chart_type: 图表类型 ('bar', 'line', 'pie', 'scatter', 'histogram')
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            **kwargs: 其他参数
            
        Returns:
            str: base64编码的图片数据
        """
        if chart_type not in self.supported_chart_types:
            raise ValueError(f"不支持的图表类型: {chart_type}. 支持的类型: {self.supported_chart_types}")
        
        # 解析数据
        parsed_data = self._parse_data(data)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 根据图表类型生成相应图表
        if chart_type == 'bar':
            self._generate_bar_chart(ax, parsed_data, title, x_label, y_label, **kwargs)
        elif chart_type == 'line':
            self._generate_line_chart(ax, parsed_data, title, x_label, y_label, **kwargs)
        elif chart_type == 'pie':
            self._generate_pie_chart(ax, parsed_data, title, **kwargs)
        elif chart_type == 'scatter':
            self._generate_scatter_chart(ax, parsed_data, title, x_label, y_label, **kwargs)
        elif chart_type == 'histogram':
            self._generate_histogram_chart(ax, parsed_data, title, x_label, y_label, **kwargs)
        
        # 调整布局
        plt.tight_layout()
        
        # 将图表转换为base64字符串
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_str
    
    def _parse_data(self, data: str) -> Dict[str, Any]:
        """
        解析查询结果数据
        
        Args:
            data: 查询结果数据（JSON格式字符串）
            
        Returns:
            Dict: 解析后的数据
        """
        try:
            # 尝试解析为JSON
            parsed_data = json.loads(data)
            return parsed_data
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试其他方式解析
            try:
                # 尝试使用ast.literal_eval解析
                import ast
                parsed_data = ast.literal_eval(data)
                return parsed_data
            except:
                # 如果都失败了，当作简单字符串处理
                return {"values": [data]}
    
    def _generate_bar_chart(self, ax, data: Dict[str, Any], title: str, 
                           x_label: str, y_label: str, **kwargs):
        """
        生成柱状图
        """
        # 处理数据
        x_data, y_data = self._extract_xy_data(data)
        
        # 绘制柱状图
        bars = ax.bar(range(len(x_data)), y_data, **kwargs)
        
        # 设置标签
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # 设置x轴刻度标签
        if len(x_data) <= 10:  # 如果数据点不多，显示所有标签
            ax.set_xticks(range(len(x_data)))
            ax.set_xticklabels(x_data, rotation=45, ha='right')
        else:  # 如果数据点多，只显示部分标签
            step = len(x_data) // 10
            ax.set_xticks(range(0, len(x_data), step))
            ax.set_xticklabels([x_data[i] for i in range(0, len(x_data), step)], 
                              rotation=45, ha='right')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    def _generate_line_chart(self, ax, data: Dict[str, Any], title: str, 
                            x_label: str, y_label: str, **kwargs):
        """
        生成折线图
        """
        # 处理数据
        x_data, y_data = self._extract_xy_data(data)
        
        # 绘制折线图
        ax.plot(range(len(x_data)), y_data, marker='o', **kwargs)
        
        # 设置标签
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # 设置x轴刻度标签
        if len(x_data) <= 10:
            ax.set_xticks(range(len(x_data)))
            ax.set_xticklabels(x_data, rotation=45, ha='right')
        else:
            step = len(x_data) // 10
            ax.set_xticks(range(0, len(x_data), step))
            ax.set_xticklabels([x_data[i] for i in range(0, len(x_data), step)], 
                              rotation=45, ha='right')
    
    def _generate_pie_chart(self, ax, data: Dict[str, Any], title: str, **kwargs):
        """
        生成饼图
        """
        # 处理数据
        labels, sizes = self._extract_xy_data(data)
        
        # 绘制饼图
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', **kwargs)
        
        # 设置标题
        ax.set_title(title)
        
        # 调整标签样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _generate_scatter_chart(self, ax, data: Dict[str, Any], title: str, 
                               x_label: str, y_label: str, **kwargs):
        """
        生成散点图
        """
        # 处理数据
        x_data, y_data = self._extract_xy_data(data)
        
        # 如果x_data是字符串，转换为数字索引
        if isinstance(x_data[0], str):
            x_indices = range(len(x_data))
            ax.scatter(x_indices, y_data, **kwargs)
            # 设置x轴刻度标签
            if len(x_data) <= 10:
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_data, rotation=45, ha='right')
            else:
                step = len(x_data) // 10
                ax.set_xticks(range(0, len(x_data), step))
                ax.set_xticklabels([x_data[i] for i in range(0, len(x_data), step)], 
                                  rotation=45, ha='right')
        else:
            ax.scatter(x_data, y_data, **kwargs)
            ax.set_xlabel(x_label)
        
        # 设置标签
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _generate_histogram_chart(self, ax, data: Dict[str, Any], title: str, 
                                 x_label: str, y_label: str, **kwargs):
        """
        生成直方图
        """
        # 处理数据
        x_data, y_data = self._extract_xy_data(data)
        
        # 如果y_data存在且有意义，则使用y_data绘制直方图
        if y_data and len(y_data) > 0:
            data_for_hist = y_data
        else:
            # 否则尝试使用x_data
            data_for_hist = x_data
            # 如果x_data是字符串，转换为长度
            if isinstance(data_for_hist[0], str):
                data_for_hist = [len(str(x)) for x in data_for_hist]
        
        # 绘制直方图
        ax.hist(data_for_hist, bins=min(20, len(data_for_hist)//2), **kwargs)
        
        # 设置标签
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _extract_xy_data(self, data: Dict[str, Any]) -> Tuple[List, List]:
        """
        从数据中提取X轴和Y轴数据
        
        Args:
            data: 解析后的数据
            
        Returns:
            Tuple[List, List]: (x_data, y_data) 元组
        """
        # 如果数据是字典形式，尝试从中提取键值对
        if isinstance(data, dict):
            # 如果有明确的键值对
            if 'keys' in data and 'values' in data:
                return data['keys'], data['values']
            # 如果是普通的键值对字典
            elif len(data) > 0:
                keys = list(data.keys())
                # 检查值是否都是数值类型
                values = list(data.values())
                if all(isinstance(v, (int, float)) for v in values):
                    return keys, values
                else:
                    # 如果值不是数值，可能需要特殊处理
                    return keys, list(range(len(keys)))
        
        # 如果数据是列表形式
        elif isinstance(data, list):
            # 如果是嵌套列表，可能是[[x1,y1], [x2,y2], ...]格式
            if len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                x_data = [item[0] for item in data]
                y_data = [item[1] for item in data]
                return x_data, y_data
            # 如果是一维列表，用索引作为x轴
            else:
                x_data = list(range(len(data)))
                y_data = data
                return x_data, y_data
        
        # 默认情况
        return ['未知'], [0]

# 单例模式
chart_generator = ChartGenerator()