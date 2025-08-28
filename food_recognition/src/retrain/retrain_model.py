"""
食材识别模型重训练脚本
在已有模型基础上进行增量训练

作者: zhangpeng
时间: 2025-08-28
"""

import os
import argparse
from ultralytics import YOLO


def retrain_model(model_path, data_config, epochs=50, imgsz=640):
    """
    重训练食材识别模型
    
    Args:
        model_path (str): 预训练模型路径
        data_config (str): 数据集配置文件路径
        epochs (int): 训练轮数
        imgsz (int): 图像尺寸
    
    Returns:
        model: 重训练后的模型
    """
    # 模型保存目录（写死方式）
    save_dir = '../../models'
    
    # 创建模型保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载已有模型
    model = YOLO(model_path)
    
    # 开始重训练
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        cache=True,  # 缓存图像以加快训练速度
        project=save_dir,  # 模型保存路径
        name='food_model_retrain'  # 模型保存名称
    )
    
    print(f"重训练模型已保存至: {os.path.join(save_dir, 'food_model_retrain')}")
    return model


def main():
    parser = argparse.ArgumentParser(description='重训练食材识别模型')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_config', type=str, default='../../data/food_dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    
    args = parser.parse_args()
    
    print("开始重训练食材识别模型...")
    model = retrain_model(args.model_path, args.data_config, args.epochs, args.imgsz)
    print("模型重训练完成并已保存!")


if __name__ == '__main__':
    main()