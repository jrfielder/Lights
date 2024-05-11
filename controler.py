import io
import time
from picamera2 import Picamera2
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
import cv2
import RPi.GPIO as GPIO
from rpi_ws281x import PixelStrip, Color
from joblib import load

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

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def apply_gaussian_smoothing(image, kernel_size, sigma):
    """Applies Gaussian smoothing using a Gaussian kernel."""
    kernel = gaussian_kernel(kernel_size, sigma)
    padded_image = np.pad(image, [(kernel.shape[0]//2, kernel.shape[0]//2), 
                                  (kernel.shape[1]//2, kernel.shape[1]//2)], 
                                  mode='constant', constant_values=0)
    result = np.zeros_like(image)

    # Convolution process
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return result

def frame_difference(prev_frame, current_frame, threshold=30):
    diff = np.abs(prev_frame - current_frame)
    diff[diff < threshold] = 0
    diff[diff >= threshold] = 255
    return diff

def compute_gradients(image):
    """ Compute the gradient image in x and y direction """
    gx = np.empty(image.shape, dtype=np.float32)
    gy = np.empty(image.shape, dtype=np.float32)
    gx[:, :-1] = np.diff(image, axis=1)  # horizontal gradient
    gy[:-1, :] = np.diff(image, axis=0)  # vertical gradient

    gx[:, -1] = 0  # edges to zero
    gy[-1, :] = 0  # edges to zero

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    return magnitude, orientation

def cell_histogram(magnitude, orientation, cell_size, bin_size):
    """ Create histograms of gradients in cells """
    bins = np.arange(0, 180+bin_size, bin_size)
    cell_mag = np.zeros((int(magnitude.shape[0] / cell_size), int(magnitude.shape[1] / cell_size), len(bins)-1))

    for i in range(cell_mag.shape[0]):
        for j in range(cell_mag.shape[1]):
            cell_m = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_o = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_hist, _ = np.histogram(cell_o, bins=bins, weights=cell_m, density=True)
            cell_mag[i, j, :] = cell_hist

    return cell_mag

def hog_features(image, cell_size=8, bin_size=20):
    """ Calculate HOG features for the whole image """
    magnitude, orientation = compute_gradients(image)
    hog_image = cell_histogram(magnitude, orientation, cell_size, bin_size)
    # Flatten to make it a single feature vector
    hog_features = hog_image.ravel()
    return hog_features


def compute_gradients(image):
    """ Compute gradients in the horizontal and vertical direction using a simple Sobel operator. """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 360
    return magnitude, orientation

def convolve(image, kernel):
    """ Apply a convolutional kernel to an image. """
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, ((kernel_height // 2, kernel_width // 2), (kernel_height // 2, kernel_width // 2)), mode='constant')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = (kernel * padded_image[i:i+kernel_height, j:j+kernel_width]).sum()
    return output

def non_maximum_suppression(magnitude, orientation):
    """ Thin edges using non-maximum suppression. """
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = orientation / 45.0  # Quantize the orientation to 8 cases (0, 1, 2, ..., 7)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            # Identify the neighbours to interpolate
            if (0 <= angle[i, j] < 1) or (7 <= angle[i, j] <= 8):
                neighbors = [magnitude[i, j+1], magnitude[i, j-1]]
            elif 1 <= angle[i, j] < 2:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 2 <= angle[i, j] < 3:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            elif 3 <= angle[i, j] < 4:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 4 <= angle[i, j] < 5:
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 5 <= angle[i, j] < 6:
                neighbors = [magnitude[i+1, j-1], magnitude[i-1, j+1]]
            elif 6 <= angle[i, j] < 7:
                neighbors = [magnitude[i+1, j], magnitude[i-1, j]]
            
            # Suppress edges that are not stronger than their neighbors
            if magnitude[i, j] >= max(neighbors):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    
    return Z

def double_threshold(image, low_threshold, high_threshold):
    """ Apply double thresholding and link edges. """
    strong = 255
    weak = 75
    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    output = np.zeros_like(image)
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
    return output, strong, weak

def hysteresis(image, strong=255, weak=75):
    """Apply hysteresis to finalize edge detection by connecting weak edges to strong edges."""
    M, N = image.shape
    # Iterate through every pixel in the image except the edge pixels
    for i in range(1, M-1):
        for j in range(1, N-1):
            if image[i, j] == weak:
                # Check the 8 neighboring pixels to see if any of them are strong
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                    or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                    or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    # If no strong edges are adjacent, suppress this pixel
                    image[i, j] = 0
    return image


def canny_edge_detector(image, sigma=1.0, low_threshold=50, high_threshold=100):
    """Apply the Canny edge detector to an image."""
    # Step 1: Noise reduction
    smoothed_image = gaussian_filter(image, sigma=sigma)
    # Step 2: Gradient calculation
    magnitude, orientation = compute_gradients(smoothed_image)
    # Step 3: Non-maximum suppression
    thin_edges = non_maximum_suppression(magnitude, orientation)
    # Step 4: Double thresholding
    thresholded_edges, strong, weak = double_threshold(thin_edges, low_threshold, high_threshold)
    # Step 5: Edge Tracking by Hysteresis
    final_edges = hysteresis(thresholded_edges, strong, weak)
    
    return final_edges

# Video handling
def capture_frames():
    # Initialize the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    while True:

        # Capture frames and pre-process them
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)

        # Run algorithms for hand detection

        # Get the HOG Features
        hog_feat = hog_features(processed_frame)
        # Get the frame difference
        frame_difference = frame_difference(prev_frame, processed_frame)
        # Get Canny edges
        canny_edges =  canny_edge_detector(processed_frame)
        # Get CNN fatures


        # Combine into one vector
        combined_features = np.concatenate([hog_feat,canny_edges.flatten(),frame_difference.flatten,])

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        # Display the frame using OpenCV
        cv2.imshow('Processed Frame', processed_frame)

        prev_frame = processed_frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def set_color_and_brightness(color, brightness):
    strip.setBrightness(brightness)
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()


# LED Info
LED_COUNT = 50     
LED_PIN = 18          
LED_FREQ_HZ = 800000 
LED_DMA = 10          
LED_BRIGHTNESS = 50  
LED_INVERT = False  
LED_CHANNEL = 0       

# create pixel strip
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
strip.begin()        


# Start capturing frames
capture_frames()