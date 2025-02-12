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
    print("检测到人脸{}", faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def main(image_path='images/many_face.jpg', cascade_path='data/haarcascades/haarcascade_frontalface_default.xml'):
    try:
        # 获取摄像头的视频流 默认写死0
        capture = cv2.VideoCapture(0)
        # 获取帧宽度、高度、fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        # 判断摄像头是否打开
        if not capture.isOpened():
            print('找不到摄像头视频流')
            exit()

        # 从摄像头读取视频，直到关闭
        while capture.isOpened():
            # 通过摄像头捕获帧
            ret, frame = capture.read()
            if not ret:
                print('找不到视频帧')
                break
            # 把捕获的帧变成灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 显示原来视频流
            cv2.imshow('frame', frame)
            # 显示灰度的视频流
            cv2.imshow('frame', gray)

            # 创建人脸识别分类器
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                print('分类器加载失败')
                return

            # 识别人脸
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            # 绘制人脸并获取结果图片
            img_face_result = plot_face(frame.copy(), faces)
            # 显示绘制完的视频流
            cv2.imshow('plotface', img_face_result)
            # 键盘敲击q，退出
            if cv2.waitKey(1) == ord('q'):
                break


        # # 创建画布
        # plt.figure(figsize=(10, 10))
        # plt.suptitle('Face Recognition', fontsize=16, fontweight='bold')
        #
        # # 显示原始图片和绘制人脸后的图片
        # show_image(img, 'Original Image', 1)
        # show_image(img_face_result, 'Face Result', 2)
        #
        # # 最后显示所有图像
        # plt.show()

        # 释放所有资源
        capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()
