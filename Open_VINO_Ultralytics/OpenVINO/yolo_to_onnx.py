import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best.pt')  # Replace with your model path

# Export the model to ONNX format
model.export(format='onnx')
