openvino:
image 1/1 /home/shagnik/Documents/Ramen_Internship/frame_00004.jpg: 640x640 22 oranges, 304.9ms
Speed: 3.3ms preprocess, 304.9ms inference, 5.2ms postprocess per image at shape (1, 3, 640, 640)


normal: 
image 1/1 /home/shagnik/Documents/Ramen_Internship/frame_00004.jpg: 640x640 22 oranges, 368.9ms
Speed: 1.8ms preprocess, 368.9ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 640)


Inference time for openVINO model is less than the original model, but due to high image pre-processing time, OpenVINO ends up taking longer than base Yolo-V8.
