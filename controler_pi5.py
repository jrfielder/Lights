import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from gpiozero import PWMLED
from time import sleep


# Preprocessing function with sigma value for testing gaussian smoothing
def process_frame(frame, sigma=1):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian smoothing
    blurred = gaussian_filter(gray, sigma=sigma)
    
    # Normalize image
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def capture_frames():
    # Initialize the USB camera
    cap = cv2.VideoCapture(0)  # Use the appropriate index if you have multiple cameras

    while True:
        # Capture frames and pre-process them
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame is not captured

        processed_frame = process_frame(frame)

        # Run algorithms for hand detection

        # cv2.imshow("Camera", frame)
        # cv2.waitKey(1)
  
        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Hand gesture detection
        # If detected mark gesture coordinates



    # Release the camera
    cap.release()
    cv2.destroyAllWindows()


def set_brightness(duty_cycle): # Duty cycle should be between 0-1
    led.value = duty_cycle

 
led = PWMLED(18)
# Start capturing frames
capture_frames()
