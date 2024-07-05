#!/usr/bin/env python3
import warnings
from super_gradients.training import models
from super_gradients.common.object_names import Models
import supervision as sv
import cv2 as cv
import numpy as np
import time

warnings.filterwarnings('ignore')

model = models.get(Models.YOLO_NAS_S, num_classes=1, checkpoint_path='yoloNAS/average_model.pth')

cap = cv.VideoCapture('/home/shagnik/Documents/Ramen_Internship/wfl.mp4')
assert cap.isOpened(), "Error: Could not open Video Capture"

byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Perform object detection
    predictions = model.predict(frame)

    end_predict_time = time.time()
    end_predict = end_predict_time - start_time
    print(f"Prediction time: {end_predict}")

    detections = sv.Detections.from_yolo_nas(predictions)
    detections = detections[detections.class_id == 0]

    # Update tracker with detections
    detections = byte_tracker.update_with_detections(detections=detections)
    # print(tracked_detections)
    
    # print(detections)
    class_names = predictions.class_names
    labels = [
        f"#{tracker_id} {class_names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]
    
    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
    
    cv.imshow("YOLO-NAS-BYTE-TRACKER", annotated_frame)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time}")

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

    