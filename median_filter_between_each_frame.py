import cv2
import numpy as np
import time

start = time.time()
# Input and output video paths
input_video = 'Podu_Ros.mp4'
output_video = 'median_between_frames_4.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter.fourcc(*'mp4v')

# Initialize the VideoWriter
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Buffer to store frames
buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the current frame to the buffer
    buffer.append(gray_frame)

    # Keep only the last two frames in the buffer
    if len(buffer) > 360:
        buffer.pop(0)

    # If we have at least 360 frames, apply temporal median filter
    if len(buffer) == 360:
        # Calculate the pixel-wise median
        median_frame = np.median(np.stack(buffer), axis=0).astype(np.uint8)
        median_frame_bgr = cv2.cvtColor(median_frame, cv2.COLOR_GRAY2BGR)
        out.write(median_frame_bgr)

stop = time.time()
cap.release()
out.release()
cv2.destroyAllWindows()
# A print for prformance check
print(f"Processing finished took {stop - start} seconds..")