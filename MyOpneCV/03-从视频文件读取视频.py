import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='视频文件路径')
args = parser.parse_args()
print(args.video_path)

# 加载视频文件
capture = cv2.VideoCapture(args.video_path)

# 获取摄像头的视频流q
capture = cv2.VideoCapture(args.video_path)

# 获取帧宽度、高度、fps
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)

# 从摄像头读取帧
ret, frame = capture.read()
while ret:
    cv2.imshow('frame', frame)
    # 再次读取帧
    ret, frame = capture.read()
    # 键盘敲击q，退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放所有资源，并关闭窗口
capture.release()
cv2.destroyAllWindows()