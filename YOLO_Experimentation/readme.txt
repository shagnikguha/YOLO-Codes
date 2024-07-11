YOLO-V8
    • High accuracy and low inference time
    • Extremely simple training process
    • Best to use
    • Lots of resources available for optimization using OpenVINO
YOLO-NAS
    • Has three model types: small, medium, and large
    • Models have high accuracy and high inference time on both GPU and CPU
    • Training is more complicated than YOLO-V8 but still relatively simple
    • Requires COCO format of data
    • Good, but due to high inference time, it is suggested to keep this as a backup
    • Low accuracy when optimized with OpenVINO and has very few resources online for OpenVINO optimization for object detection
YOLO-X
    • Still in development
    • No downloadable package; GitHub repo has to be downloaded
    • Training code has several issues at the source (fixes marked out in training notebook)
    • Accuracy is average at best and takes a lot of time for inference compared to YOLO-NAS and YOLO-V8
    • Not working with supervision
    • Due to all this, it is best not to use
YOLO-V10
    • New to the market
    • Hardly detects anything after training for 30 epochs
    • Detects objects after training for 10 epochs, but object detection flickers after each frame
    • Better to use YOLO-V8
Licensing Issues
    • YOLO-V8/V10: All Ultralytics models require a license if we want to keep our source code private
    • YOLO-NAS: All SuperGradient models require a license if we decide to use their model weights. However, it seems that they are following suit with Ultralytics
    • YOLO-X: Free to use but not worth it
