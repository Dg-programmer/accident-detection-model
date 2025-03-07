import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import os
from PIL import Image


# Load the model
interpreter = Interpreter(model_path="C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\Unplugged 2.0\\accident_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details to check expected shape and type
print("Input details:", input_details)
print("Expected input shape:", input_details[0]['shape'])

# Path to your test images
test_dir = "C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\accident dataset\\data\\test"  # Update with your test image folder

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Resize to the model input size (10, 5)
    expected_height = input_details[0]['shape'][1]
    expected_width = input_details[0]['shape'][2]
    
    img = img.resize((expected_width, expected_height))  # Resize to model input size
    
    # Convert image to grayscale (as the model expects a 2D matrix, no channels)
    img = img.convert("L")  # Convert to grayscale
    
    # Convert to numpy array, and normalize the pixel values
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (Shape: [1, 10, 5])
    img = img / 255.0  # Normalize if needed
    
    return img

# Loop through the test images and get predictions
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    
    # Check if the file is an image
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Check image shape before passing to interpreter
        print("Image shape before passing to interpreter:", image.shape)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor (probabilities for each class)
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Print the raw output (probabilities)
        print(f"Raw output for {image_name}: {output}")
        
        # For classification, you can use np.argmax to get the predicted class
        output_class = np.argmax(output)
        print(f"Prediction for {image_name}: Class {output_class}")
