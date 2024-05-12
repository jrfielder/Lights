import cv2
import numpy as np
import os
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.ndimage import gaussian_filter
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load a pre-trained VGG16 model without the classifier layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def process_frame(frame, sigma=1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, sigma=sigma)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_vgg = preprocess_input(image.astype('float32'))
    features_cnn = base_model.predict(np.expand_dims(image_vgg, axis=0)).flatten()

    # For color images, ensure you are using the right axis for skimage
    # Here, convert BGR to RGB since skimage expects images in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # HOG features for color image
    features_hog, _ = hog(image_rgb, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    gray_image = process_frame(image)  # still using grayscale for Canny
    edges = cv2.Canny(gray_image, 100, 200)
    features_canny = edges.flatten()

    return np.concatenate([features_cnn, features_hog, features_canny])

def load_data(directory, label):
    data, labels = [], []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                features = extract_features(filepath)
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    return np.array(data), np.array(labels)

# Path to the directories
gesture_dir = "Lights\\gestures\\thumb_up\\frames"
nongesture_dir = "Lights\\gestures\\no_gestures\\frames"


# Load data
gesture_features, gesture_labels = load_data(gesture_dir, 1)
nongesture_features, nongesture_labels = load_data(nongesture_dir, 0)

# Combine data
features = np.concatenate([gesture_features, nongesture_features])
labels = np.concatenate([gesture_labels, nongesture_labels])

# Normalize features
scaler = Normalization()
scaler.adapt(features)

# Model setup
model = Sequential([
    Dense(1, activation='linear', kernel_initializer='he_normal')
])
model.compile(optimizer='adam', loss='hinge')

# Split data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
model.fit(scaler(x_train), y_train, epochs=10, validation_data=(scaler(x_test), y_test))

# Evaluate the model on test data
test_loss = model.evaluate(scaler(x_test), y_test)
predictions = (model.predict(scaler(x_test)) > 0).astype(int)
accuracy = np.mean(predictions.flatten() == y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")



# Saving the model in HDF5 format
model.save('my_model.h5')


# Loading the model from HDF5 file
loaded_model = tf.keras.models.load_model('my_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
