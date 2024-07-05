import cv2
import numpy as np
from super_gradients.training.utils.media.image import load_image
import onnxruntime
import time
# import matplotlib.pyplot as plt
import cv2 as cv
# from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization

DETECTION_CLASSES_LIST = ['orange']

def show_predictions_from_flat_format(image, predictions):
    [flat_predictions] = predictions

    image = image.copy()
    class_names = DETECTION_CLASSES_LIST
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        image = DetectionVisualization.draw_box_title(
                    image_np=image,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_index,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )

    # plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    # plt.tight_layout()
    # plt.show()
    cv.imshow('result', image)
    cv.waitKey(0)

image = load_image("/home/shagnik/Documents/Ramen_Internship/frame_00004.jpg")
image = cv2.resize(image, (640, 640))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession("/home/shagnik/Documents/Ramen_Internship/yoloNAS/optimized/yolonas_orange_s.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
start = time.perf_counter()
result = session.run(outputs, {inputs[0]: image_bchw})
end = time.perf_counter()
print(f"Time: {end-start} ms")

print(result)

flat_predictions = result
for (_, x_min, y_min, x_max, y_max, confidence, class_id) in flat_predictions[0]:
    class_id = int(class_id)
    print(f"Detected object with class_id={class_id}, confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")


# print(COCO_DETECTION_CLASSES_LIST)
 
show_predictions_from_flat_format(image=image, predictions=flat_predictions)