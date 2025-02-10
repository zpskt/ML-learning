import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('camera_id', type=int, default=0, help='camera ID')
args = parser.parse_args()
print(args.camera_id)

# 获取摄像头的视频流
capture = cv2.VideoCapture(args.camera_id)

# 获取帧宽度、高度、fps
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)
# 打印出来这些数据查看
print(f'width: {width}, height: {height}, fps: {fps}')
# 判断摄像头是否打开
if not capture.isOpened():
    print('Error opening video stream or file')
    exit()
# 从摄像头读取视频，直到关闭
while capture.isOpened():
    # 通过摄像头捕获帧
    ret, frame = capture.read()
    if not ret:
        print('Can\'t receive frame (stream end?). Exiting ...')
        break
    # 把捕获的帧变成灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 显示原来视频流
    cv2.imshow('frame', frame)
    # 显示灰度的视频流
    cv2.imshow('frame', gray)
    # 键盘敲击q，退出
    if cv2.waitKey(1) == ord('q'):
        break
# 释放所有资源，并关闭窗口
capture.release()
cv2.destroyAllWindows()