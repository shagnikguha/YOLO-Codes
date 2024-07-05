from ultralytics import YOLO
# import openvino as ov

model = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best.pt')
ov_model  = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best_openvino_model/')

results = ov_model('/home/shagnik/Documents/Ramen_Internship/frame_00004.jpg')

result_original = model('/home/shagnik/Documents/Ramen_Internship/frame_00004.jpg')