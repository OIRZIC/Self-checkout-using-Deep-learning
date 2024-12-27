import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

import torch

from byte_tracker.tracker import byte_tracker
import time

""" Import YOLO in ONNX format"""
# Model source
model_path = r"D:\DO AN TOT NGHIEP\DATASET\STATIONARY DATA\self-made-v7\best_training_output_v8s.onnx"
# Webcam source
video_source  = 0
if not os.path.isfile(model_path):
    raise ValueError("Please set the `onnx_model_path` variable to the correct value")

EP_list = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
]
session = ort.InferenceSession(model_path,providers=EP_list)

# Class list
class_names = [
    "BUT_CHI_DIXON",
    "BUT_HIGHLIGHT_MNG_TIM",
    "BUT_HIGHLIGHT_RETRO_COLOR",
    "BUT_LONG_SHARPIE_XANH",
    "BUT_NUOC_CS_8623",
    "BUT_XOA_NUOC",
    "HO_DOUBLE_8GM",
    "KEP_BUOM_19MM",
    "KEP_BUOM_25MM",
    "NGOI_CHI_MNG_0.5_100PCS",
    "SO_TAY_A6",
    "THUOC_CAMPUS_15CM",
    "THUOC_DO_DO",
    "THUOC_PARABOL",
    "XOA_KEO_CAPYBARA_9566"
]

#Onnx parameter
score_th = 0.3
nms_th = 0.45

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

if __name__ == "__main__":
    yolo_model_path = r"D:\DO AN TOT NGHIEP\DATASET\STATIONARY DATA\self-made-v7\best_training_output_v8s.onnx"
    YOLO_detector = YOLO(model=yolo_model_path)
    # Setup video Source
    cap = cv2.VideoCapture(video_source)
    # Read Source
    capture = cv2.VideoCapture(video_source)

    frame_id = 0
    skip_frame_rate = 2  # Skip every 2 frames, can be adjusted


    while True:
        start_time = time.time()

        ret, frame = capture.read()

        if not ret:
            break
        else:
            frame_id +=1
            if frame_id % skip_frame_rate !=0:
                continue

            frame = cv2.resize(frame, (640,480))
            orig_frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]
            frame_size = np.array([frame_h, frame_w])
            cv2.imshow("detect",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
