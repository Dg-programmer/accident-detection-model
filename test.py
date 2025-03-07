import os
import numpy as np

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image

# Path to the directory containing test images
test_dir = 'C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\accident dataset\\data\\test'  # Replace with your test images folder

# Load the TFLite model
interpreter = Interpreter(model_path="C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\Unplugged 2.0\\accident_detection_model.tflite")  # Replace with your .tflite model path
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess and make predictions
def make_prediction(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Replace with your input image size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if needed
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Iterate over the images in the test directory and make predictions
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        print(f"Processing image: {img_name}")
        
        # Get prediction
        output_data = make_prediction(img_path)
        
        # For binary classification
        prediction = "Class 1" if output_data[0][0] > 0.5 else "Class 0"
        print(f"Predicted Class for {img_name}: {prediction}")
