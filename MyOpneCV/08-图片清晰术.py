import cv2
import numpy as np

# 读取图像
image = cv2.imread('path_to_your_image.jpg')

# 将图像从BGR转换为RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换到HSV颜色空间，以便更容易地处理颜色
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# 定义天空的颜色范围（例如，蓝色和白色）
lower_blue = np.array([90, 100, 100])  # 蓝色范围低值
upper_blue = np.array([110, 255, 255])  # 蓝色范围高值
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# 使用形态学操作来填充和细化天空区域
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)
mask = cv2.erode(mask, kernel, iterations=1)

# 创建一个蓝色的天空颜色
sky_blue = np.array([110, 255, 255])  # 蓝色值可以根据需要调整
image_hsv[mask > 0] = sky_blue

# 将HSV图像转换回RGB
sky_blue_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

# 显示结果
cv2.imshow('Sky Blue Image', sky_blue_image)
cv2.waitKey(0)
cv2.destroyAllWindows()