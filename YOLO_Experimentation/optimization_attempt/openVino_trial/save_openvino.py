import openvino as ov
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

# path to an image file
source = '/home/shagnik/Documents/Ramen_Internship/frame_00004.jpg'
# loading an image for testing YOLO-NAS Pose model.
model = models.get(Models.YOLO_NAS_S, 
                   num_classes=1,
                   checkpoint_path='yoloNAS/average_model.pth')
# Testing model on loaded image
model.predict(source, conf=0.50).show()
# prepare dummy input data
input_data = torch.rand(1, 3, 224, 224)
# converting model to Openvino
ov_model = ov.convert_model(model, example_input=input_data)
# save model to OpenVINO IR for later use
ov.save_model(ov_model, 'model.xml')
# Compile and infer with OpenVINO:
compiled_model = ov.compile_model('model.xml')
# run inference
result = compiled_model(input_data)
print(result)