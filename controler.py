import io
import time
import picamera2
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
import cv2


# Preprocessing function with sigma value for testing gaussian smoothing
def process_frame(frame, sigma=1):
    # grayscalePpp
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
    with picamera2.PiCamera() as camera:
        camera.framerate = 30
        camera.resolution = (640, 480)
        time.sleep(2) 
        
		# initialize stream
        stream = picamera2.array.PiRGBArray(camera, size=(640, 480))
        
        for _ in camera.capture_continuous(stream, format="bgr", use_video_port=True):
            # Convert the image to a NumPy array and process it
            frame = stream.array
            processed_frame = process_frame(frame)
            
            # Display the frame using OpenCV
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Clear the stream in preparation for the next frame
            stream.truncate(0)
        
        cv2.destroyAllWindows()

# Start capturing frames
capture_frames()