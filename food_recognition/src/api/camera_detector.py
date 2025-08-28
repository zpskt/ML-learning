"""
实时摄像头食材识别模块
调用笔记本摄像头进行实时图像分类识别食材

作者: zhangpeng
时间: 2025-08-28
"""

import cv2
from ultralytics import YOLO
import argparse


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
    
    def start_detection(self):
        """
        启动摄像头实时检测
        """
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("摄像头已启动，按 'q' 键退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 使用模型进行预测
            results = self.model(frame)
            
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
                            
                            # 绘制边界框和标签
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{cls_name} {confidence:.2f}', 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.9, (0, 255, 0), 2)
            
            # 显示画面
            cv2.imshow('Food Detection', frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
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