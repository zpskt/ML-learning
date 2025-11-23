#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的图表功能测试脚本，生成并保存图表到文件
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_agent.chart_generator import ChartGenerator
import json
import base64

def test_and_save_charts():
    """
    测试图表生成功能并保存为文件
    """
    print("开始测试图表生成功能...")
    
    # 创建图表生成器实例
    generator = ChartGenerator()
    
    # 测试数据1: 简单的键值对数据
    test_data1 = json.dumps({
        "产品A": 100,
        "产品B": 150,
        "产品C": 80,
        "产品D": 200
    })
    
    # 创建保存图表的目录
    if not os.path.exists("test_charts"):
        os.makedirs("test_charts")
    
    # 测试柱状图
    print("生成柱状图...")
    try:
        bar_chart_base64 = generator.generate_chart(
            data=test_data1,
            chart_type='bar',
            title='产品销量柱状图',
            x_label='产品',
            y_label='销量'
        )
        
        # 保存为文件
        bar_chart_data = base64.b64decode(bar_chart_base64)
        with open("test_charts/bar_chart.png", "wb") as f:
            f.write(bar_chart_data)
        print("柱状图已保存到 test_charts/bar_chart.png")
    except Exception as e:
        print(f"生成柱状图时出错: {e}")
    
    # 测试折线图
    print("生成折线图...")
    try:
        line_chart_base64 = generator.generate_chart(
            data=test_data1,
            chart_type='line',
            title='产品销量趋势',
            x_label='产品',
            y_label='销量'
        )
        
        # 保存为文件
        line_chart_data = base64.b64decode(line_chart_base64)
        with open("test_charts/line_chart.png", "wb") as f:
            f.write(line_chart_data)
        print("折线图已保存到 test_charts/line_chart.png")
    except Exception as e:
        print(f"生成折线图时出错: {e}")
    
    # 测试饼图
    print("生成饼图...")
    try:
        pie_chart_base64 = generator.generate_chart(
            data=test_data1,
            chart_type='pie',
            title='产品销量占比'
        )
        
        # 保存为文件
        pie_chart_data = base64.b64decode(pie_chart_base64)
        with open("test_charts/pie_chart.png", "wb") as f:
            f.write(pie_chart_data)
        print("饼图已保存到 test_charts/pie_chart.png")
    except Exception as e:
        print(f"生成饼图时出错: {e}")
    
    # 测试数据2: 列表形式数据
    test_data2 = json.dumps([
        ["一月", 120],
        ["二月", 135],
        ["三月", 150],
        ["四月", 140],
        ["五月", 160]
    ])
    
    # 测试散点图
    print("生成散点图...")
    try:
        scatter_chart_base64 = generator.generate_chart(
            data=test_data2,
            chart_type='scatter',
            title='月份销售散点图',
            x_label='月份',
            y_label='销售额'
        )
        
        # 保存为文件
        scatter_chart_data = base64.b64decode(scatter_chart_base64)
        with open("test_charts/scatter_chart.png", "wb") as f:
            f.write(scatter_chart_data)
        print("散点图已保存到 test_charts/scatter_chart.png")
    except Exception as e:
        print(f"生成散点图时出错: {e}")
    
    # 测试数据3: 数值列表
    test_data3 = json.dumps([10, 20, 30, 25, 15, 35, 40])
    
    # 测试直方图
    print("生成直方图...")
    try:
        hist_chart_base64 = generator.generate_chart(
            data=test_data3,
            chart_type='histogram',
            title='数值分布直方图',
            x_label='数值',
            y_label='频率'
        )
        
        # 保存为文件
        hist_chart_data = base64.b64decode(hist_chart_base64)
        with open("test_charts/histogram_chart.png", "wb") as f:
            f.write(hist_chart_data)
        print("直方图已保存到 test_charts/histogram_chart.png")
    except Exception as e:
        print(f"生成直方图时出错: {e}")
    
    print("\n图表生成功能测试完成。所有图表已保存到 test_charts 目录中。")
    print("你可以打开这些PNG文件查看生成的图表。")

if __name__ == "__main__":
    test_and_save_charts()