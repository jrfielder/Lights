import cv2
import numpy as np
import os
from skimage.feature import hog
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from gpiozero import PWMLED
import tflite_runtime.interpreter as tflite


def process_frame(frame, sigma=1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, sigma=sigma)
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


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
    # Assuming the frame height is 480 pixels, and we want -1 to 1 mapping 224
    pwm_value = dy / (224/2)  # Scale and shift
    pwm_value = np.clip(pwm_value, -1, 1)  # Ensure within range
    # Convert to a suitable PWM duty cycle (0 to 1)
    duty_cycle = (pwm_value + 1) / 2
    led.value = duty_cycle


# Function to load and prepare the image
def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    image = cv2.resize(image, (224, 224))
    
    # Convert the pixel values to float32
    image = image.astype('float32')
    
    # Subtract the mean pixel value of the training dataset
    mean_pixel = np.array([103.939, 116.779, 123.68])  # Mean pixel values of ImageNet dataset
    image -= mean_pixel
    
    # Scale the pixel values to be in the range [-1, 1]
    image /= 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)



def capture_frames(interpreter):
    print('here0')
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
    print('here1')
    # prev_frame = process_frame(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    print('here')
    print(prediction)
    if prediction==False:
        # processed_frame = process_frame(frame)
        # hog_feat = hog_features(processed_frame)
        # frame_diff_result = frame_compare(prev_frame, processed_frame)
        # canny_edges = cv2.Canny(processed_frame, 80, 150)  # Using OpenCV Canny function

        # # combined_features = np.concatenate([hog_feat, canny_edges.flatten(), frame_diff_result.flatten()])
        # combined_features = extract_features(frame)

        # Prepare your input data (this needs to match the training preprocessing)
        input_data = preprocess_image(frame)

        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run the interpreter
        interpreter.invoke()
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Get the output predictions from the model
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        # update if frame has gesture
        prediction = output_data
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

# Assign the LED GPIO
led = PWMLED(18)

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='/home/john/Desktop/lights/Lights/model.tflite')

# Initialize a simple tracker, here using a dummy variable
tracker = cv2.TrackerKCF_create()

# Flag to denote when the gesture is being tracked
tracking_active = False

capture_frames(interpreter)
