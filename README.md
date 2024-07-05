# YOLO-Codes

## Overview of Files
├── README.md # Overview of files
├── YOLO_Experimentation # Folder containing code for several YOLO models
│ ├── Experimentation_Results.docx # Contains summarized findings and observations regarding the various models used
│ ├── wfl.mp4 # The video on which everything has been tested
│ ├── training_notebooks # Contains detailed notebooks for training YOLO models
│ │ ├── YOLO_NAS_Training.ipynb # YOLO-NAS training notebook
│ │ ├── yolov8training.ipynb # YOLOv8 training notebook
│ │ ├── yolov10_training.ipynb # YOLOv10 training notebook
│ │ └── YOLOX_training.ipynb # YOLO-X training notebook
│ ├── processing_time # Contains CSV files with inference times for each frame in the video on both CPU and GPU
│ │ ├── processing_time_cpu_yolov8.csv # CSV for YOLOv8 on CPU
│ │ ├── processing_time_gpu_yolov8.csv # CSV for YOLOv8 on GPU
│ │ ├── processing_times_cpu_yolonas.csv # CSV for YOLO-NAS on CPU
│ │ └── processing_times_gpu_yolonas.csv # CSV for YOLO-NAS on GPU
│ ├── optimization_attempt # Files related to YOLO-NAS optimization
│ │ ├── optimized
│ │ │ ├── yolonas_orange_s.onnx # YOLO-NAS model in ONNX format
│ │ │ ├── yolonas-onnx.py # Code to convert YOLO-NAS to ONNX
│ │ │ └── run_onnx.py # Code to run ONNX model
│ │ └── openVino_trail # Contains code for OpenVINO optimization (didn't work)
│ └── codes # Code for different YOLO models
│ ├── yolo_NAS # Code for YOLO-NAS
│ │ ├── average_model.pth # YOLO-NAS small model weights
│ │ ├── Live_Video_Processing.py # Runs YOLO-NAS on video in real-time
│ │ ├── model_predict # Basic code to run YOLO-NAS
│ │ ├── Real_time_Byte_tracker.py # Runs YOLO-NAS with Byte Tracker from the Supervision library
│ │ └── webcam.py # Runs YOLO-NAS on webcam
│ └── yolov8 # Code for YOLOv8
│ ├── best.pt # Weights for trained YOLOv8 model
│ ├── object_count.py # Code for object counting (video is processed first, then shown)
│ ├── realtime_object_detection.py # Real-time object tracking
│ ├── region_counting.py # Counting objects passing through a defined region
│ └── counting_via_line.py # Counting objects crossing a defined line
└── Open_VINO_Ultralytics # Folder containing optimization code for YOLOv8 using OpenVINO package
├── note.txt # Contains experimentation results of optimized and non-optimized model
├── best_openvino_model # Contains the optimized IR model using OpenVINO
└── OpenVINO # Contains code for YOLO conversion to ONNX and IR model implementation
├── yolo_to_onnx.py # Code to convert YOLO model to ONNX model
├── onnx_verify.py # Code to confirm proper ONNX conversion
├── yolo_to_IR.py # Code to convert YOLO to OpenVINO IR model
└── VINO_run.py # Code to run IR model
