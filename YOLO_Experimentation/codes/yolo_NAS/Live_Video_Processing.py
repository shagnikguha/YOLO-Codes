import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
import supervision as sv
import numpy as np

model = models.get(Models.YOLO_NAS_S, num_classes=1, checkpoint_path='yoloNAS/average_model.pth')

cap = cv2.VideoCapture("/home/shagnik/Documents/Ramen_Internship/wfl.mp4")
# cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error: Could not open video capture."

box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictions = model.predict(rgb_frame)

    detections = sv.Detections.from_yolo_nas(predictions)
    detections = detections[detections.class_id == 0]

    class_names = predictions.class_names

    labels = [f"{class_names[class_id]} {confidence:0.2f}" for (bbox, _, confidence, class_id, _, _) in detections]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    cv2.imshow("YOLO-NAS Real-time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
