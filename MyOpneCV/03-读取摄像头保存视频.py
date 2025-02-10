import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('camera_id', type=int, default=0, help='camera ID')
parser.add_argument('output_file', type=str, default='output.avi', help='output file')
args = parser.parse_args()

# 获取摄像头的视频流
capture = cv2.VideoCapture(args.camera_id)

# 判断摄像头是否打开
if not capture.isOpened():
    print('Error opening video stream or file')
    exit()
# 获取帧的属性
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)
# 对视频进行编码
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 初始化视频写入对象
# 参数1: 输出文件的路径和名称
# 参数2: 视频编解码器，例如 'XVID' 或 'MJPG'
# 参数3: 视频的帧率
# 参数4: 视频帧的大小，为一个元组，包含宽度和高度
video_writer = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# 读取摄像头
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
    video_writer.write(gray)
    # 键盘敲击q，退出
    if cv2.waitKey(1) == ord('q'):
        break
# 释放所有资源，并关闭窗口
capture.release()
video_writer.release()
cv2.destroyAllWindows()