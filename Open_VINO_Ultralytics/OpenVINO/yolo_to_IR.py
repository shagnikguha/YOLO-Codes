from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best.pt')

model.export(format='openvino')

