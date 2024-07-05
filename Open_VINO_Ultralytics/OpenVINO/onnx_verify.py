import onnx

# Load the ONNX model
onnx_model = onnx.load('/home/shagnik/Documents/Ramen_Internship/yolov8_working/OpenVINO/best.onnx')

# Check the model
onnx.checker.check_model(onnx_model)

print('The model is successfully converted to ONNX format.')
