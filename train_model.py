import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

# Generate dummy data (replace with your actual dataset)
x_train = np.random.rand(1000, 10, 5).astype(np.float32)  # 1000 samples, 10 timesteps, 5 features
y_train = np.random.randint(0, 2, (1000, 1)).astype(np.float32)  # Binary labels

# Define model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, 5)),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Save Keras model
model.save("accident_detection_model.h5")

# Convert to TensorFlow Lite (TFLite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable built-in TFLite operations
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops if needed
]
converter._experimental_lower_tensor_list_ops = False  # Fix TensorList issue
converter.experimental_enable_resource_variables = True

try:
    tflite_model = converter.convert()
    with open("accident_detection_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("TFLite model conversion successful!")
except Exception as e:
    print("TFLite conversion failed:", e)
