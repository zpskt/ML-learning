import cv2
import matplotlib.pyplot as plt
import numpy as np

# 设置字体颜色常量字典
colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 0, 255),
    'magenta': (255, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}


def show_image(img, title):
    # 顺序转换：BGR -> RGB
    img_rgb = img[:, :, ::-1]
    # 显示标题
    plt.title(title)
    plt.imshow(img_rgb)
    plt.show()


# 创建一个400x400像素的黑色画布，用于后续绘制操作。该画布是一个三维数组，每个像素有三个通道（RGB），数据类型为8位无符号整数。
canvas = np.zeros((400, 400, 3), np.uint8)
# 将画布填充为白色。canvas.fill(255) 中的参数 255 表示使用白色（在灰度模式下，255 表示最亮的白色）填充整个画布。
canvas.fill(255)
'''
在坐标(100, 100)处绘制文本“Hello World”
使用字体cv2.FONT_HERSHEY_SIMPLEX，字体大小为1
文本颜色为红色，线条粗细为2像素
'''
cv2.putText(canvas, 'Hello World', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['red'], 2)

if __name__ == '__main__':
    # 调用方法展示
    show_image(canvas, 'Text on Black Canvas')
