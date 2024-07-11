# YOLO-Codes Overview

## Repository Contents

This repository contains detailed training notebooks for YOLOv8, YOLOv10, YOLO-NAS, and YOLO-X. It also includes codes to run YOLO and YOLO-NAS for real-time detection and tracking.

## Overview of Files

- YOLO_Experimentation
  - Experimentation_Results.docx
  - wfl.mp4
  - training_notebooks
    - YOLO_NAS_Training.ipynb
    - yolov8training.ipynb
    - yolov10_training.ipynb
    - YOLOX_training.ipynb
  - processing_time
    - processing_time_cpu_yolov8.csv
    - processing_time_gpu_yolov8.csv
    - processing_times_cpu_yolonas.csv
    - processing_times_gpu_yolonas.csv
  - optimization_attempt
    - optimized
      - yolonas_orange_s.onnx
      - yolonas-onnx.py
      - run_onnx.py
    - openVino_trail
  - codes
    - yolo_NAS
      - average_model.pth
      - Live_Video_Processing.py
      - model_predict
      - Real_time_Byte_tracker.py
      - webcam.py
    - yolov8
      - best.pt
      - object_count.py
      - realtime_object_detection.py
      - region_counting.py
      - counting_via_line.py
- Open_VINO_Ultralytics
  - note.txt
  - best_openvino_model
  - OpenVINO
    - yolo_to_onnx.py
    - onnx_verify.py
    - yolo_to_IR.py
    - VINO_run.py
- RT_DETR
  - RT_DETR.ipynb
  - first_11_frames(1).mp4
  - processing_times_RT_DETR_gpu.csv
  - readme.txt
