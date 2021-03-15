import cv2
import time
import numpy as np

video_capture = cv2.VideoCapture(0)

print('Start stream..')
prevTime = 0
while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)  # resize frame (optional)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()