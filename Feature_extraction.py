import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.ndimage import gaussian_filter

# Load a pre-trained VGG16 model without the classifier layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Preprocessing function with sigma value for testing gaussian smoothing
def process_frame(frame, sigma=1):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian smoothing
    blurred = gaussian_filter(gray, sigma=sigma)
    
    # Normalize image
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

# Function to extract features from an image
def extract_features(image_path):
    # Load image and preprocess for VGG16
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_vgg = preprocess_input(image.astype('float32'))

    # CNN features
    features_cnn = base_model.predict(np.expand_dims(image_vgg, axis=0)).flatten()

    # preprocess for canny and hog
    gray_image = process_frame(image)
	
    # HOG features
    features_hog, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    # Canny Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)
    features_canny = edges.flatten()

    # Combine features
    return np.concatenate([features_cnn, features_hog, features_canny])



# Get features from data of gesture and no gesture

extract_features


# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# SVM
# Assume `features` and `labels` are your training data and labels
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', use_bias=True)
])

# Compile the model using hinge loss which is standard for SVM
model.compile(optimizer='adam', loss='hinge')

# Train the model
model.fit(features, labels, epochs=10)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('svm_model.tflite', 'wb') as f:
    f.write(tflite_model)
