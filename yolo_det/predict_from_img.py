#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：predict_from_img.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/19 09:28 
@Description： 
'''
from food_recognition.src.api.app import model_path

if __name__ == '__main__':
    from ultralytics import YOLO
    import cv2  # 用于显示和保存图像

    model_path = '/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/runs/detect/train-8/weights/best.pt'
    # Load a pretrained YOLO26n model
    model = YOLO(model_path)
    # Define path to the image file
    source = "/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/datasets/Vegetable-Object-Detection/test/images/0a1b802a-fruit_apple_2025-07-29T13-58-54-535Z_033.jpg"

    # Run inference on the source
    results = model.predict(source,show=True)  # list of Results objects
    for result in results:
        ...
        result.save_crop(save_dir="./crops", file_name="detection")

    # --- 绘制和保存结果 ---
    # 1. 获取绘制了检测结果的图像（BGR格式，方便OpenCV处理）
    plotted_image = results[0].plot()  # results[0] 是单张图片的结果

    # 2. 显示图像（可选）
    cv2.imshow("YOLO Detection Result", plotted_image)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()

    # 3. 保存图像到当前目录
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, plotted_image)
    print(f"检测结果已保存至: {output_path}")
