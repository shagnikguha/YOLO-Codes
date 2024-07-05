import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

object_counter = 0

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    global object_counter

    results = model(frame, imgsz=1280)[0]

    detections = sv.Detections.from_ultralytics(results)

    detections = detections[detections.class_id == 0]

    # print(detections)
    triggered = zone.trigger(detections=detections)
    # if triggered.any():
    #     object_counter += 1

    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    
    # for detection in detections:
    #     print(detection)
    
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for (bbox, _, confidence, class_id, _, _) in detections]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    cv2.putText(frame, f'Total Objects: {object_counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

video_path = '/home/shagnik/Downloads/temp/wfl.mp4'

polygon = np.array([
    [336, 550],[438, 515],[630, 611],[492, 709]
])

video_info = sv.VideoInfo.from_video_path(video_path)

zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.blue(), thickness=2, text_thickness=1, text_scale=0.5)

model = YOLO('/home/shagnik/Downloads/temp/best.pt')

sv.process_video(source_path=video_path, target_path='result.mp4', callback=process_frame)
