import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import numpy as np

# Function to preprocess the dataset
def preprocess_data(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),  # Resize images to (128, 128) for the model
        batch_size=32,
        class_mode='binary'  # Binary classification (accident or not)
    )

    # Flow validation images in batches
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

# Function to create the CNN-LSTM model for accident detection
def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Output layer: binary classification (accident or not)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )
    return history

# Function to save the model
def save_model(model):
    model.save("accident_detection_model.h5")
    print("Model saved successfully!")

# Function to convert the model to TensorFlow Lite
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("accident_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("TensorFlow Lite model conversion successful!")

# Main function
def main():
    train_dir = 'C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\accident dataset\\data\\train'  # Replace with the path to your training data directory
    validation_dir = 'C:\\Users\\gdhru\\OneDrive\\Desktop\\Coding\\accident dataset\\data\\val'  # Replace with the path to your validation data directory
    
    # Step 1: Preprocess the data
    train_generator, validation_generator = preprocess_data(train_dir, validation_dir)
    
    # Step 2: Create the model
    model = create_model()
    
    # Step 3: Train the model
    train_model(model, train_generator, validation_generator)
    
    # Step 4: Save the model
    save_model(model)
    
    # Step 5: Convert the model to TensorFlow Lite format
    convert_to_tflite(model)

if __name__ == '__main__':
    main()
