import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('/home/shagnik/Documents/Ramen_Internship/yolov8_working/best.pt')

# Open video file
cap = cv2.VideoCapture('/home/shagnik/Documents/Ramen_Internship/wfl.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = 'output.avi' 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame in the video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    detections = model(frame)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 255, 0)

        # Drawing bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        class_name = model.names[class_id]

        cv2.putText(frame, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

    cv2.imshow('Result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
