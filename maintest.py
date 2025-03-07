import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter # Correct import for TensorFlow Lite Interpreter
import numpy as np
import os
from PIL import Image

# Load the model
interpreter = Interpreter(model_path="C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\Unplugged 2.0\\accident_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Path to your test images
test_dir = "C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\accident dataset\data\\test"  # Update with your test image folder

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize the image if needed (ensure this matches your model's expected input size)
    img = np.array(img)  # Convert image to numpy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image (ensure this matches your model's preprocessing)
    return img

# Loop through the test images and get predictions
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Print the prediction result
    print(f"Prediction for {image_name}: {output}")
