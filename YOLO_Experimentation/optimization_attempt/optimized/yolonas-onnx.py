from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.conversion import DetectionOutputFormatMode
from super_gradients.conversion.conversion_enums import ExportQuantizationMode
from super_gradients.conversion import ExportTargetBackend

model = models.get(Models.YOLO_NAS_S, 
                   num_classes=1,
                   checkpoint_path='yoloNAS/average_model.pth')


export_result = model.export(
    "yolonas_orange_s.onnx",
    confidence_threshold = 0.1,
    nms_threshold = 0.5,
    num_pre_nms_predictions = 100,
    max_predictions_per_image = 50,
    output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT,
    #engine=ExportTargetBackend.TENSORRT
)

print(export_result)