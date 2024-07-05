import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

model = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best.pt')

LINE_START = sv.Point(438, 515)
LINE_END = sv.Point(630, 611)

video_path = '/home/shagnik/Documents/Ramen_Internship/wfl.mp4'

def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(color=sv.Color.red(), thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    for result in model.track(source=video_path, show=True, stream=True, agnostic_nms=True):
        frame = result.orig_img  # Correct attribute for the original image
        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.class_id == 0]

        labels = [f"{model.names[class_id]} {confidence:0.2f}" for (_, _, confidence, class_id, _, _) in detections]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        line_counter.trigger(detections=detections)

        frame = line_annotator.annotate(frame=frame, line_counter=line_counter)

        cv2.imshow('yolov8', frame)

        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
