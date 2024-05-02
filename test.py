import cv2

# Initialize the camera capture object with the USB camera source
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set properties, example to set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
