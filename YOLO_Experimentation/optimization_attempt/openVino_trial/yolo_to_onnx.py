from super_gradients.training import models
from super_gradients.common.object_names import Models

net = models.get(Models.YOLO_NAS_S, 
                   num_classes=1,
                   checkpoint_path='yoloNAS/average_model.pth')

models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="/home/shagnik/Documents/Ramen_Internship/yoloNAS/int8/yolo_nas_s.onnx")