#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML-learning 
@File    ：predict_from_stream.py
@IDE     ：PyCharm 
@Author  ：张鹏
@Date    ：2026/7/19 09:28 
@Description： 
'''
if __name__ == '__main__':
    import cv2

    from ultralytics import YOLO

    model_path = '/Users/zhangpeng/Desktop/workspace/github/ML-learning/yolo_det/runs/detect/train-8/weights/best.pt'
    # Load a pretrained YOLO26n model
    model = YOLO(model_path)
    # Open the video file
    # video_path = "path/to/your/video/file.mp4"
    # cap = cv2.VideoCapture(video_path)
    # Open the default camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model(frame,conf = 0.8 )

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
