#!/usr/bin/env python3.9
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
from gpiozero import PWMLED
from tensorflow.keras.models import load_model

# Load a pre-trained VGG16 model without the classifier layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def process_frame(frame, sigma=1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, sigma=sigma)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def adjust_brightness_based_on_movement(initial_pos, current_pos):
    dy = current_pos[1] - initial_pos[1]  # Change in y position
    # Scale dy to a suitable range for PWM adjustment, e.g., -1 to 1
    # Assuming the frame height is 480 pixels, and we want -1 to 1 mapping 224
    pwm_value = dy / (224/2)  # Scale and shift
    pwm_value = np.clip(pwm_value, -1, 1)  # Ensure within range
    # Convert to a suitable PWM duty cycle (0 to 1)
    duty_cycle = (pwm_value + 1) / 2
    led.value = duty_cycle


def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image_vgg = preprocess_input(image.astype('float32'))
    features_cnn = base_model.predict(np.expand_dims(image_vgg, axis=0)).flatten()
    # print("CNN ",features_cnn.shape)
    # For color images, ensure you are using the right axis for skimage
    # Here, convert BGR to RGB since skimage expects images in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # HOG features for color image
    features_hog, _ = hog(image_rgb, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    # print("HOG ",features_hog.shape)
    gray_image = process_frame(image)  # still using grayscale for Canny
    edges = cv2.Canny(gray_image, 100, 200)
    features_canny = edges.flatten()
    # print("canny ",features_canny.shape)
    final_feature = np.concatenate([features_cnn, features_hog, features_canny])
    # print(final_feature.shape)
    return final_feature

def capture_frames():

    # Load the model
    # Assign the LED GPIO
    led = PWMLED(18)

    
    prediction = False
    duty_cycle = 0.3  # Initial duty cycle (0-1) - light set to off
    led.value=duty_cycle
    # initilize timer
    frame_timer=0
    """ Captures frames from camera, processes them, and displays results. """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture initial frame.")
        cap.release()
        cv2.destroyAllWindows()
        return
    # prev_frame = process_frame(frame)
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        print(prediction)
        if prediction==False:
            # processed_frame = process_frame(frame)
            # hog_feat = hog_features(processed_frame)
            # frame_diff_result = frame_compare(prev_frame, processed_frame)
            # canny_edges = cv2.Canny(processed_frame, 80, 150)  # Using OpenCV Canny function

            # # combined_features = np.concatenate([hog_feat, canny_edges.flatten(), frame_diff_result.flatten()])
            # combined_features = extract_features(frame)

            # Prepare your input data (this needs to match the training preprocessing)
            features = extract_features(frame)
            # print("Feature ",features.shape)
            reshaped_input = np.reshape(features, (1, 101508))
            # pred = model.predict(reshaped_input)
            # print("pred ",pred)
            # Normalize features
            # Normalize features
            scaler = Normalization()
            scaler.adapt(features)
            # # Assuming 'features' is the input data
            # scaled_features = scaler(features)  # Scale the features
            # predictions = (pred > 0).astype(int)  # Make predictions

            prediction = (model.predict(scaler(reshaped_input)) > 0).astype(int)
            # cv2.imshow("Camera", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            print(prediction)

            # preped_vector = prepare_input(combined_features,input_details)
            # prediction = predict_gesture(preped_vector, interpreter, input_details, output_details)
            if prediction:
                bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                # Track gesture
                tracker.init(frame, bbox)
                tracking_active = True
                initial_pos = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                
            ## Gesture detected allow light control 
        else:
            success, bbox = tracker.update(frame)

            if success:
                current_pos = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                # Change duty cycle
                adjust_brightness_based_on_movement(initial_pos,current_pos)

            frame_timer+=frame_timer

            if frame_timer == 120:
                prediction = False
            # # Display Canny
            # canny_edges_3channel = np.stack((canny_edges,)*3, axis=-1)
            # combined_frame = np.hstack((frame, canny_edges_3channel))



        # prev_frame = processed_frame
    cap.release()
    cv2.destroyAllWindows()


# Initialize a simple tracker, here using a dummy variable
tracker = cv2.TrackerKCF_create()
model = load_model('/home/john/Desktop/lights/Lights/my_model.h5')
# Flag to denote when the gesture is being tracked
tracking_active = False

capture_frames()
