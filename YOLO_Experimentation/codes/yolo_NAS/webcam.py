import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import  Models

model = models.get('yolo_nas_s', num_classes= 1, checkpoint_path='yoloNAS/average_model.pth')

model.predict_webcam()
