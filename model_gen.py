import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM, Dropout
from tensorflow.keras.models import Sequential


# Parameters
video_height, video_width = 224, 224  # Size to which each video frame will be resized
max_frames = 30  # Maximum number of frames per video to process
import os
import cv2
import numpy as np

def load_videos(directory, label):
    videos = []
    labels = []
    print("Loading videos from:", directory)
    for video_file in os.listdir(directory):
        video_path = os.path.join(directory, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (video_width, video_height))
                frames.append(frame)
        finally:
            cap.release()
        videos.append(frames)
        labels.append(label)  # Assign the label for each video
    return videos, labels

# Example usage, assuming directories are set up for positive and negative examples
positive_videos, positive_labels = load_videos(os.path.expanduser("Lights\\gestures\\thumb_up"), 1)
negative_videos, negative_labels = load_videos(os.path.expanduser("Lights\\gestures\\no_gestures"), 0)

# Combine positive and negative examples
videos = positive_videos + negative_videos
labels = positive_labels + negative_labels

