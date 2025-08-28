"""
实时摄像头食材识别模块
调用笔记本摄像头进行实时图像分类识别食材

作者: zhangpeng
时间: 2025-08-28
"""

import cv2
from ultralytics import YOLO
import argparse
from typing import List
import sys
import os
import logging


class CameraFoodDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        初始化摄像头食材识别器
        
        Args:
            model_path (str): 模型文件路径
            conf_threshold (float): 置信度阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.cap = None
        
        # 尝试使用配置管理器设置日志
        self._setup_logging()
        
        # 初始化事件处理机制
        try:
            # 添加项目根目录到Python路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # 动态导入事件处理模块
            import importlib.util
            event_handler_path = os.path.join(project_root, 'src', 'events', 'event_handler.py')
            spec = importlib.util.spec_from_file_location("event_handler", event_handler_path)
            event_handler_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(event_handler_module)
            
            self.event_handler = event_handler_module.event_handler
            self.FoodDetectionEvent = event_handler_module.FoodDetectionEvent
            
        except Exception as e:
            self.event_handler = None
            logging.error(f"警告: 事件处理模块初始化失败，将不触发事件: {e}")
    
    def _setup_logging(self):
        """设置日志记录"""
        try:
            from src.config.config_manager import config_manager
            config_manager.setup_logging()
        except ImportError:
            # 如果无法导入配置管理器，使用默认日志设置
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            import logging
            from logging.handlers import RotatingFileHandler
            
            # 设置日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # 设置文件处理器
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, "camera_detector.log"),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            
            # 获取记录器并添加处理器
            logger = logging.getLogger("camera_detector")
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            
            # 同时也为root logger添加处理器，确保所有日志都被记录
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            if not any(isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers):
                root_logger.addHandler(file_handler)
    
    def start_detection(self):
        """
        启动摄像头实时检测
        """
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logging.error("无法打开摄像头")
            return
        
        logging.info("摄像头已启动，按 'q' 键退出")
        print("摄像头已启动，按 'q' 键退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("无法读取摄像头画面")
                break
            
            # 使用模型进行预测
            results = self.model(frame)
            
            # 存储当前帧检测到的食材
            detected_ingredients: List[str] = []
            
            # 在画面中绘制检测结果
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取置信度
                        confidence = box.conf[0].item()
                        if confidence > self.conf_threshold:
                            # 获取类别名称和边界框坐标
                            cls_id = int(box.cls[0].item())
                            cls_name = self.model.names[cls_id]
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # 添加到检测到的食材列表
                            if cls_name not in detected_ingredients:
                                detected_ingredients.append(cls_name)
                            
                            # 绘制边界框和标签
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name} {confidence:.2f}', 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.9, (0, 255, 0), 2)
            
            # 如果检测到食材且事件处理器可用，则触发事件
            if detected_ingredients and self.event_handler:
                try:
                    event = self.FoodDetectionEvent(
                        ingredients=detected_ingredients,
                        source="camera"
                    )
                    self.event_handler.trigger_event(event)
                    logging.info(f"检测到食材: {detected_ingredients}")
                except Exception as e:
                    logging.error(f"触发事件时出错: {e}")
            
            # 显示画面
            cv2.imshow('Food Detection', frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("用户退出摄像头检测")
                break
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """
        析构函数，确保释放摄像头资源
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='实时摄像头食材识别')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 创建检测器实例并启动检测
    detector = CameraFoodDetector(args.model_path, args.conf_threshold)
    detector.start_detection()


if __name__ == '__main__':
    main()