"""
食材识别API服务
提供视频食材识别接口

作者: zhangpeng
时间: 2025-08-28
"""

import os
import cv2
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np


app = Flask(__name__)

# 全局模型变量
model = None
model_path = None


def load_model(model_file_path):
    """
    加载模型
    
    Args:
        model_file_path (str): 模型文件路径
    
    Returns:
        model: 加载的模型对象
    """
    global model, model_path
    model = YOLO(model_file_path)
    model_path = model_file_path
    return model


def detect_ingredients_in_video(video_path, conf_threshold=0.5):
    """
    在视频中检测食材
    
    Args:
        video_path (str): 视频文件路径
        conf_threshold (float): 置信度阈值
    
    Returns:
        list: 检测到的食材列表
    """
    # todo: 实现视频检测逻辑
    return []
    
def detect_ingredients_in_image(image_path, conf_threshold=0.5):
    """
    在图片中检测食材
    
    Args:
        image_path (str): 图片文件路径
        conf_threshold (float): 置信度阈值
    
    Returns:
        list: 检测到的食材列表
    """
    global model
    
    if model is None:
        raise ValueError("模型未加载，请先调用load_model函数加载模型")
    
    # 读取图片
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("无法读取图片文件")
    
    # 使用模型进行预测
    results = model(image)
    
    # 存储检测到的食材
    detected_ingredients = set()
    
    # 解析检测结果
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 获取置信度
                confidence = box.conf[0].item()
                if confidence > conf_threshold:
                    # 获取类别名称
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    detected_ingredients.add(cls_name)
    
    return list(detected_ingredients)
    cap = cv2.VideoCapture(video_path)
    
    # 存储检测到的食材
    detected_ingredients = set()
    
    frame_count = 0
    # 处理视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 每隔30帧处理一次，提高处理速度
        if frame_count % 30 == 0:
            # 使用模型进行预测
            results = model(frame)
            
            # 解析检测结果
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取置信度
                        confidence = box.conf[0].item()
                        if confidence > conf_threshold:
                            # 获取类别名称
                            cls_id = int(box.cls[0].item())
                            cls_name = model.names[cls_id]
                            detected_ingredients.add(cls_name)
        
        frame_count += 1
    
    cap.release()
    
    return list(detected_ingredients)


@app.route('/')
def index():
    """首页"""
    return jsonify({
        "message": "食材识别API服务",
        "endpoints": {
            "/load_model": "加载模型",
            "/detect": "检测视频中的食材",
            "/detect_image": "检测图片中的食材"
        }
    })


@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """加载模型接口"""
    data = request.get_json()
    model_file_path = data.get('model_path')
    
    if not model_file_path or not os.path.exists(model_file_path):
        return jsonify({"error": "模型文件路径无效"}), 400
    
    try:
        load_model(model_file_path)
        return jsonify({"message": "模型加载成功"})
    except Exception as e:
        return jsonify({"error": f"模型加载失败: {str(e)}"}), 500


@app.route('/detect', methods=['POST'])
def detect_ingredients():
    """检测视频中的食材"""
    data = request.get_json()
    video_path = data.get('video_path')
    conf_threshold = data.get('conf_threshold', 0.5)
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "视频文件路径无效"}), 400
    
    try:
        ingredients = detect_ingredients_in_video(video_path, conf_threshold)
        return jsonify({
            "message": "检测完成",
            "ingredients": ingredients
        })
    except Exception as e:
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


@app.route('/detect_image', methods=['POST'])
def detect_ingredients_in_image_endpoint():
    """检测图片中的食材"""
    data = request.get_json()
    image_path = data.get('image_path')
    conf_threshold = data.get('conf_threshold', 0.5)
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "图片文件路径无效"}), 400
    
    try:
        ingredients = detect_ingredients_in_image(image_path, conf_threshold)
        return jsonify({
            "message": "检测完成",
            "ingredients": ingredients
        })
    except Exception as e:
        return jsonify({"error": f"检测失败: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)