import cv2
import numpy as np


def edit_picture():
    # 读取图片
    img = cv2.imread('images/corrosion.png')
    # 检查图像是否成功加载
    if img is None:
        print(f"无法加载图像: images/corrosion.png")
        return
    # 显示图片
    cv2.imshow('img', img)
    # 等待
    cv2.waitKey(0)
    # 关闭图片
    # 关闭所有之前打开的窗口
    cv2.destroyAllWindows()

    # 创建一个5x5的全1数组作为卷积核
    kernel = np.ones((5, 5), np.uint8)

    # 对图像进行腐蚀处理
    erode = cv2.erode(img, kernel, iterations=1)

    # 显示腐蚀处理后的图像，等待用户按键后关闭窗口
    cv2.imshow('erode', erode)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 读取图像文件
    img = cv2.imread('images/corrosion.png')

    # 显示原图，等待用户按键后关闭窗口
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 再次创建一个5x5的全1数组作为卷积核
    kernel = np.ones((5, 5), np.uint8)

    # 对图像进行不同程度的腐蚀处理
    erosion_1 = cv2.erode(img, kernel, iterations=1)
    erosion_2 = cv2.erode(img, kernel, iterations=2)
    erosion_3 = cv2.erode(img, kernel, iterations=3)

    # 将三次腐蚀处理的结果并排显示
    res = np.hstack([erosion_1, erosion_2, erosion_3])

    # 显示腐蚀处理后的图像，等待用户按键后关闭窗口
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    edit_picture()
