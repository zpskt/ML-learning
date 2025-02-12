import cv2
import matplotlib.pyplot as plt


# 显示图片
def show_image(img, title, position):
    # bgr -> rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.subplot(2, 2, position)
    plt.imshow(img_rgb)
    plt.axis('off')


# 绘制人脸
def plot_face(img, faces):
    if not faces.any() :
        print("未检测到人脸")
        return img
    print("检测到人脸{}", faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def main(image_path='images/many_face.jpg', cascade_path='data/haarcascades/haarcascade_frontalface_default.xml'):
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print('图片不存在')
            return

        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 创建人脸识别分类器
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print('分类器加载失败')
            return

        # 识别人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        # 绘制人脸并获取结果图片
        img_face_result = plot_face(img.copy(), faces)

        # 创建画布
        plt.figure(figsize=(10, 10))
        plt.suptitle('Face Recognition', fontsize=16, fontweight='bold')

        # 显示原始图片和绘制人脸后的图片
        show_image(img, 'Original Image', 1)
        show_image(img_face_result, 'Face Result', 2)

        # 最后显示所有图像
        plt.show()

        # 释放所有资源
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()
