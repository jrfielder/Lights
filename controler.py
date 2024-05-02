import io
import time
from picamera2 import Picamera2
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
import cv2


# Preprocessing function with sigma value for testing gaussian smoothing
def process_frame(frame, sigma=1):
    # grayscale
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Gaussian smoothing
    blurred = gaussian_filter(gray, sigma=sigma)
    
    # Normalize image
    normalized = (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred))
    normalized = np.ascontiguousarray((normalized * 255).astype(np.uint8))
    return normalized


# Video handling
def capture_frames():
    # Initialize the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    while True:
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)


        # Display the frame using OpenCV
        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

# Start capturing frames
capture_frames()