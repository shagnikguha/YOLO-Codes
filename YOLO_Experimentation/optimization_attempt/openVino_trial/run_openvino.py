import openvino as ov
import numpy as np
import cv2 as cv

def preprocess_image(image_path, target_shape):
    # Load the image
    image = cv.imread(image_path)
    
    # Resize the image to the target height and width
    target_height, target_width = target_shape[2], target_shape[3]
    image_resized = cv.resize(image, (target_width, target_height))
    
    # Convert the image to a floating point tensor and normalize pixel values if necessary
    image_normalized = image_resized.astype(np.float32)
    
    # Change the data layout from HWC to CHW
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    
    # Add a batch dimension
    image_batch = np.expand_dims(image_transposed, axis=0)
    
    return image_batch

def draw_bounding_boxes(image_path, output):
    
    image = cv.imread(image_path)
    
    for bbox in output[0]:
        x_min, y_min, x_max, y_max = bbox
        # Draw the bounding box
        cv.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    return image

# Initialize the OpenVINO Runtime
core = ov.Core()

# Read the IR model
model = core.read_model('/home/shagnik/Documents/Ramen_Internship/yoloNAS/int8/model.xml')

# Compile the model for a specific device (e.g., CPU)
compiled_model = core.compile_model(model, 'CPU')

# Example input preprocessing
# input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Define the path to your image
image_path = '/home/shagnik/Documents/Ramen_Internship/frame_00004.jpg'

# Define the target shape
target_shape = (1, 3, 224, 224)

input_data = preprocess_image(image_path, target_shape)

# Create an inference request
infer_request = compiled_model.create_infer_request()

# Perform inference
result = infer_request.infer({0: input_data})

# Access the output
output = result[compiled_model.output(0)]
print(output)

result_img = draw_bounding_boxes(image_path, output)

cv.imshow('result', result_img)

cv.waitKey(0)