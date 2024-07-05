import openvino.runtime as ov

ov_model = ov.convert_model('/home/shagnik/Documents/Ramen_Internship/yoloNAS/int8/yolo_nas_s.onnx')

ov.save_model(ov_model, '/home/shagnik/Documents/Ramen_Internship/yoloNAS/int8/yolo_nas_s.xml')