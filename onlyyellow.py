import cv2
import numpy as np
from PIL import Image
from utilme import det_limits

# Define the yellow color range
yellow = [0, 255, 255]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper limits for yellow color
    lowerlimit, upperlimit = det_limits(color=yellow)

    # Create mask to detect only yellow objects
    mask = cv2.inRange(hsvImage, lowerlimit, upperlimit)

    # Apply mask to extract only yellow parts
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert mask to image for bounding box detection
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    # Draw bounding box around detected yellow object
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Show the processed frame
    cv2.imshow('Yellow Object Detection', result)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
