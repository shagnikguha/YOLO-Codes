from super_gradients.training import models
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
MODEL_ARCH = 'yolo_nas_l'

best_model = models.get(
    MODEL_ARCH,
    num_classes=1,
    checkpoint_path="/home/shagnik/Downloads/temp/average_model.pth"
).to(DEVICE)

input_video_path = "/home/shagnik/Downloads/temp/wfl.mp4"
output_video_path = "detections_rerun.mp4"
#device=0

best_model.predict(input_video_path).save(output_video_path)
