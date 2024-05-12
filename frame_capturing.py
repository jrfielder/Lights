# Import necessary libraries
import os
import cv2

# Parameters
video_height, video_width = 224, 224  # Size to which each video frame will be resized

def save_video_frames(input_directory, output_directory, label):
    print("Saving video frames from:", input_directory)
    for video_file in os.listdir(input_directory):
        video_path = os.path.join(input_directory, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (video_width, video_height))
                frame_filename = f"{os.path.splitext(video_file)[0]}_frame{frame_count}_label{label}.jpg"
                cv2.imwrite(os.path.join(output_directory, frame_filename), frame)
                frame_count += 1
        finally:
            cap.release()

# Example usage, assuming directories are set up for positive and negative examples and output directories exist
save_video_frames(os.path.expanduser("Lights\\gestures\\thumb_up"), os.path.expanduser("Lights\\gestures\\thumb_up\\frames"), 1)
save_video_frames(os.path.expanduser("Lights\\gestures\\no_gestures"), os.path.expanduser("Lights\\gestures\\no_gestures\\frames"), 0)
