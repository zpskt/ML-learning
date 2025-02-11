import cv2
import numpy as np
import matplotlib.pyplot as plt
def show_image(img,title,position):
    # 顺序转换：BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 显示标题
    plt.subplot(position)
    plt.title(title)
    plt.imshow(img_rgb)
# 显示直方图
def show_histogram(img,title,position,color):
    # 计算图像的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 绘制直方图
    plt.subplot(position)
    plt.title(title)
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    # 范围
    plt.xlim([0, 256])
    plt.plot(hist, color=color)
def main():
    # 创建画布
    plt.figure(figsize=(10, 5))
    plt.suptitle('Histogram', fontsize=16, fontweight='bold')
    # 加载图片
    img = cv2.imread('images/color-img.png')
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 显示图片
    show_image(img, 'Original Image', 121)
    # 显示灰度图
    show_image(gray, 'Gray Image', 122)
    # 显示直方图
    show_histogram(gray, 'Gray Histogram', 221, 'r')

if __name__ == '__main__':
    main()
    plt.show()