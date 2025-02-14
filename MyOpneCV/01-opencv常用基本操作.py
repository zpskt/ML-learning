import cv2
import matplotlib.pyplot as plt

'''
彩色图片操作示例
'''


def read_color_picture():
    # 读取图片
    img = cv2.imread('images/color-img.png')
    # 获取图片形状 （返回长row、宽heights、通道channels）
    shape = img.shape
    print("图片形状为： ", shape)
    print(img.shape)
    # 获取图片大小(返回row*heights*channels)
    size = img.size
    print("图片大小为： ", size)
    # 图片类型
    dtype = img.dtype
    print(dtype)
    # opencv中，图片的排序为（B,G,R）
    # 拿到某个像素点的bgr
    (b, g, r) = img[6, 40]
    print(b, g, r)
    # 单独取某个像素点的蓝色
    b = img[6, 40, 0]
    print(b)

    # 重新给像素点复制，更换颜色
    img[6, 40] = (0, 0, 255)  # 变成红色
    # 显示图片
    cv2.imshow('img', img)
    # 等待
    cv2.waitKey(0)
    # 关闭图片
    cv2.destroyAllWindows()


'''
灰色图片操作示例
'''


def read_gray_picture():
    # 读取图片
    img = cv2.imread('images/gray-img.png', cv2.IMREAD_GRAYSCALE)
    # 获取图片形状 （返回长row、宽heights、通道channels）
    shape = img.shape
    print("图片形状为： ", shape)
    print(img.shape)
    # 获取图片大小(返回row*heights*channels)
    size = img.size
    print("图片大小为： ", size)
    # 图片类型
    dtype = img.dtype
    print(dtype)
    # opencv中，图片的排序为（B,G,R）
    # 拿到某个像素点
    print(img[6, 40])

    # 显示图片
    cv2.imshow('img', img)
    # 等待
    cv2.waitKey(0)
    # 关闭图片
    cv2.destroyAllWindows()


def read_bgr():
    import cv2
    import matplotlib.pyplot as plt
    import os

    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'images', 'color-img.png')

    try:
        # 尝试读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image file not found at {image_path}")

        # 转换 BGR 到 RGB
        image_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(img.shape)

        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 显示原始图片
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示转换后的图片
        axes[1].imshow(image_new)
        axes[1].set_title('Converted Image')
        axes[1].axis('off')

        # 展示图片
        plt.show()
        plt.close()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

# 对图片进行像素扩大化：小图变大图
def enlarge_image():

    # 读取图像
    image = cv2.imread('images/pengge.jpg')

    # 获取图像的原始尺寸
    height, width = image.shape[:2]

    # 设置新的尺寸，例如将图像放大到原来的2倍
    new_width = int(width * 10)
    new_height = int(height * 10)

    # 使用cv2.resize()进行图像缩放
    # 第三个参数是插值方法，这里使用cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # # 显示结果=
    # cv2.imshow('Resized Image', resized_image)
    # cv2.waitKey(0)
    cv2.imwrite('images/penggeBig.jpg', resized_image)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enlarge_image()
