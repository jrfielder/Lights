import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from gpiozero import PWMLED
import tflite_runtime.interpreter as tflite



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

def frame_compare(prev_frame, current_frame, threshold=30):
    diff = np.abs(prev_frame - current_frame)
    diff[diff < threshold] = 0
    diff[diff >= threshold] = 255
    return diff

def compute_gradients_simple(image):
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
    magnitude, orientation = compute_gradients_simple(image)
    hog_image = cell_histogram(magnitude, orientation, cell_size, bin_size)
    # Flatten to make it a single feature vector
    hog_features = hog_image.ravel()
    return hog_features


def compute_gradients(image):
    """Compute the image gradients using Sobel filters."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    gx = cv2.filter2D(image, -1, sobel_x)
    gy = cv2.filter2D(image, -1, sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)
    return magnitude, orientation % 180

def non_maximum_suppression(magnitude, orientation):
    """Apply non-maximum suppression to thin the edges."""
    M, N = magnitude.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = orientation / 45.0
    angle = np.round(angle) % 4

    for i in range(1, M-1):
        for j in range(1, N-1):
            neighbor_1 = 255
            neighbor_2 = 255

            # Horizontal edge
            if angle[i,j] == 0:
                neighbor_1 = magnitude[i, j+1]
                neighbor_2 = magnitude[i, j-1]
            # Diagonal edge
            elif angle[i,j] == 1:
                neighbor_1 = magnitude[i+1, j-1]
                neighbor_2 = magnitude[i-1, j+1]
            # Vertical edge
            elif angle[i,j] == 2:
                neighbor_1 = magnitude[i+1, j]
                neighbor_2 = magnitude[i-1, j]
            # Other diagonal edge
            elif angle[i,j] == 3:
                neighbor_1 = magnitude[i-1, j-1]
                neighbor_2 = magnitude[i+1, j+1]

            if magnitude[i,j] >= neighbor_1 and magnitude[i,j] >= neighbor_2:
                Z[i,j] = magnitude[i,j]
            else:
                Z[i,j] = 0

    return Z

def double_threshold(image, low_ratio, high_ratio):
    """Apply double threshold to determine strong and weak edges."""
    high_threshold = np.max(image) * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = image.shape
    res = np.zeros((M,N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

def hysteresis(image, weak=75, strong=255):
    """Apply hysteresis to finalize the edge detection."""
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if image[i, j] == weak:
                # Check the 8 neighboring cells
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong) or
                    (image[i, j-1] == strong) or (image[i, j+1] == strong) or
                    (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detector(image, low_threshold_ratio=0.1, high_threshold_ratio=0.5):
    """Full Canny edge detection process."""
    magnitude, orientation = compute_gradients(image)
    thin_edges = non_maximum_suppression(magnitude, orientation)
    thresholded_edges = double_threshold(thin_edges, low_threshold_ratio, high_threshold_ratio)
    final_edges = hysteresis(thresholded_edges)
    return final_edges

def prepare_input(feature_vector, input_details):
    # Reshape and type-cast the feature vector to match the model's input
    input_shape = input_details[0]['shape']
    input_data = np.reshape(feature_vector, input_shape).astype('float32')
    return input_data

def predict_gesture(feature_vector, interpreter, input_details, output_details):
    # Prepare the input data
    input_data = prepare_input(feature_vector, input_details)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run the inference
    interpreter.invoke()
    
    # Retrieve and return the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def adjust_brightness_based_on_movement(initial_pos, current_pos):
    dy = current_pos[1] - initial_pos[1]  # Change in y position
    # Scale dy to a suitable range for PWM adjustment, e.g., -1 to 1
    # Assuming the frame height is 480 pixels, and we want -1 to 1 mapping
    pwm_value = dy / 240.0  # Scale and shift
    pwm_value = np.clip(pwm_value, -1, 1)  # Ensure within range
    # Convert to a suitable PWM duty cycle (0 to 1)
    duty_cycle = (pwm_value + 1) / 2
    led.value = duty_cycle

def capture_frames(interpreter):

    interpreter.allocate_tensors()

    # Get the model's input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    prediction = False
    duty_cycle = 0  # Initial duty cycle (0-1) - light set to off

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

    prev_frame = process_frame(frame)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

        if prediction==False:
            processed_frame = process_frame(frame)
            hog_feat = hog_features(processed_frame)
            frame_diff_result = frame_compare(prev_frame, processed_frame)
            canny_edges = cv2.Canny(processed_frame, 80, 150)  # Using OpenCV Canny function

            combined_features = np.concatenate([hog_feat, canny_edges.flatten(), frame_diff_result.flatten()])

            preped_vector = prepare_input(combined_features,input_details)
            prediction = predict_gesture(preped_vector, interpreter, input_details, output_details)

            ## Gesture detected allow light control 
        else:

            # Change duty cycle
            adjust_brightness_based_on_movement(initial_pos,current_pos)

            frame_timer+=frame_timer
            # # Display Canny
            # canny_edges_3channel = np.stack((canny_edges,)*3, axis=-1)
            # combined_frame = np.hstack((frame, canny_edges_3channel))

            # cv2.imshow("Camera and Canny Frame", combined_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            prev_frame = processed_frame
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Assign the LED GPIO
led = PWMLED(18)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="path_to_your_model.tflite")


capture_frames(interpreter)
