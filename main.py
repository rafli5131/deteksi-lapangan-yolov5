import cv2
import torch
import numpy as np
import pathlib
from collections import deque
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('yolov5', 'custom', path='model/lapangan.pt', source='local')

video_path = "source/source-video-2.mp4"
cap = cv2.VideoCapture(video_path)
prev_detections = deque(maxlen=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = results.pred[0]

    lapangan_detections = detections[detections[:, 5] == 1]
    prev_detections.append(lapangan_detections)

    if len(prev_detections) > 0:
        smoothed_detections = np.vstack(prev_detections)

        mask = np.zeros_like(frame)
        for detection in smoothed_detections:
            x1, y1, x2, y2, confidence = detection[:5]
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)

        result = cv2.bitwise_and(frame, mask)

        cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()