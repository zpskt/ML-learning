"""
数据处理模块
用于处理图片数据、创建标签文件以及拆分训练集和验证集

作者: zhangpeng
时间: 2025-08-28
"""

import os
import random
import shutil
from pathlib import Path
import argparse


def create_dataset_structure(base_dir):
    """
    创建数据集目录结构
    
    Args:
        base_dir (str): 基础目录路径
    """
    # 创建训练集和验证集目录
    train_img_dir = os.path.join(base_dir, 'train', 'images')
    train_label_dir = os.path.join(base_dir, 'train', 'labels')
    val_img_dir = os.path.join(base_dir, 'val', 'images')
    val_label_dir = os.path.join(base_dir, 'val', 'labels')
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"创建数据集目录结构在: {base_dir}")
    return train_img_dir, train_label_dir, val_img_dir, val_label_dir


def get_food_classes(food_dir):
    """
    获取食材类别列表
    
    Args:
        food_dir (str): 食材数据根目录
    
    Returns:
        list: 食材类别列表
    """
    classes = []
    for item in os.listdir(food_dir):
        item_path = os.path.join(food_dir, item)
        if os.path.isdir(item_path):
            classes.append(item)
    
    classes.sort()
    return classes


def create_yaml_config(data_dir, classes, output_path):
    """
    创建YOLO格式的数据集配置文件
    
    Args:
        data_dir (str): 数据集根目录
        classes (list): 类别列表
        output_path (str): 配置文件输出路径
    """
    nc = len(classes)
    class_names = [f"'{cls}'" for cls in classes]
    class_names_str = ', '.join(class_names)
    
    yaml_content = f"""path: {data_dir}
train: train/images
val: val/images

names:
"""
    
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"创建数据集配置文件: {output_path}")


def process_food_data(food_dir, output_dir, val_split=0.2):
    """
    处理食材数据，生成YOLO格式的数据集
    
    Args:
        food_dir (str): 原始食材数据目录
        output_dir (str): 输出目录
        val_split (float): 验证集比例
    """
    # 创建数据集结构
    train_img_dir, train_label_dir, val_img_dir, val_label_dir = create_dataset_structure(output_dir)
    
    # 获取食材类别
    classes = get_food_classes(food_dir)
    print(f"找到 {len(classes)} 个食材类别: {classes}")
    
    # 为每个类别创建标签并复制图像
    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(food_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # 获取该类别的所有图片
        images = []
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(file)
        
        print(f"处理类别 '{class_name}'，共 {len(images)} 张图片")
        
        # 随机打乱图片列表
        random.shuffle(images)
        
        # 计算验证集数量
        val_count = int(len(images) * val_split)
        
        # 分配验证集和训练集
        val_images = images[:val_count]
        train_images = images[val_count:]
        
        print(f"  训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")
        
        # 处理训练集图片
        for image_name in train_images:
            # 复制图片
            src_img_path = os.path.join(class_dir, image_name)
            dst_img_path = os.path.join(train_img_dir, f"{class_name}_{image_name}")
            shutil.copy2(src_img_path, dst_img_path)
            
            # 创建标签文件 (YOLO格式: class_id center_x center_y width height)
            # 由于我们只需要分类，不需要检测位置，这里创建一个占位标签
            # 假设整个图片都是该类别的食材
            label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(train_label_dir, f"{class_name}_{label_name}")
            
            with open(label_path, 'w') as f:
                f.write(label_content)
        
        # 处理验证集图片
        for image_name in val_images:
            # 复制图片
            src_img_path = os.path.join(class_dir, image_name)
            dst_img_path = os.path.join(val_img_dir, f"{class_name}_{image_name}")
            shutil.copy2(src_img_path, dst_img_path)
            
            # 创建标签文件
            label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(val_label_dir, f"{class_name}_{label_name}")
            
            with open(label_path, 'w') as f:
                f.write(label_content)
    
    # 创建数据集配置文件
    yaml_path = os.path.join(output_dir, 'food_dataset.yaml')
    create_yaml_config(output_dir, classes, yaml_path)
    
    print(f"\n数据处理完成!")
    print(f"训练集图片数: {len(os.listdir(train_img_dir))}")
    print(f"验证集图片数: {len(os.listdir(val_img_dir))}")
    print(f"配置文件路径: {yaml_path}")


def main():
    # 硬编码数据目录和输出目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    food_dir = os.path.join(project_root, 'data', 'food')  # 修改为实际的数据目录路径
    output_dir = os.path.join(project_root, 'data')  # 输出到data目录
    val_split = 0.2
    
    if not os.path.exists(food_dir):
        print(f"错误: 输入目录 {food_dir} 不存在")
        return
    
    process_food_data(food_dir, output_dir, val_split)


if __name__ == '__main__':
    main()