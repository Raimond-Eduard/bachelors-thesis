import json

import cv2 as cv
import numpy as np
import time
from classes.position_tracker import EnhancedCentroidTracker
from classes.constants import *

def warm_up_mog(capture, background_substractor, warmup_frames=0):
    """
    :param capture: Video read with cv.VideoCapture
    :param background_substractor: Background substractor initialized with cv.BackgroundSubstractorMOG2()
    :param warmup_frames: If warmup frames value is 0, warm_up on all the frames the video has
    :return:
    """
    capture.set(cv.CAP_PROP_POS_FRAMES, 0)
    if warmup_frames > 0:
        for _ in range(warmup_frames):
            ret, warm_frame = capture.read()
            if not ret:
                break
            background_substractor.apply(warm_frame, learningRate=-1)
    else:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            background_substractor.apply(frame, learningRate=-1)
    return background_substractor

tracker = EnhancedCentroidTracker(max_disappeared=6, max_distance=850)
total_cars = set()
# Paths to the input videos
video_path = 'Inputs/stabilization_with_videostab_4.avi'
background_path = 'Inputs/median_between_frames_4.mp4'

# Initialize video readers
cap = cv.VideoCapture(video_path)
bg_cap = cv.VideoCapture(background_path)

# Video writer
o_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
o_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
o_fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter.fourcc(*'mp4v')
out = cv.VideoWriter('../../Outputs/podu_ros_tracking_with_stats.mp4', fourcc, o_fps, (o_width, o_height))

# Read the static background frame from the first frame of the no car video
ret_bg, background = bg_cap.read()
if not ret_bg:
    print("Failed to read background frame.")
    exit()

# Convert background to grayscale
background_gray = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

# Initialize MOG2 background subtractor
mog2 = cv.createBackgroundSubtractorMOG2(varThreshold=10, detectShadows=False)
# Count the time it takes for mog to learn the whole background
start = time.time()
mog2 = warm_up_mog(cap, mog2)

print(f"Mog took {time.time() - start} seconds to warm up.")

bg_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Region of interest
h, w = background_gray.shape[:2]

roi = np.zeros((h, w), dtype=np.uint8)

cv.fillPoly(roi, [included_region_3], (255, 255, 255))
cv.fillPoly(roi, [excluded_small_white_blob_left], (0, 0 ,0 ))
cv.fillPoly(roi, [excluding_false_detection], (0,0,0))

times_per_frame = {}

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Moving objects via MOG2
    fg_mask = mog2.apply(frame)
    # Clean noise in MOG2 mask
    fg_mask = cv.medianBlur(fg_mask, 7)
    _, fg_mask = cv.threshold(fg_mask, 127, 255, cv.THRESH_BINARY)

    dark_mask = cv.inRange(frame_gray, 35, 70)
    dark_mask = cv.morphologyEx(dark_mask, cv.MORPH_OPEN, (7,7))
    dark_mask = cv.dilate(dark_mask, (5,5), iterations=2)
    # Stationary objects via absdiff
    blurred_frame = cv.GaussianBlur(frame_gray, (5,5), 0)
    blurred_bg = cv.GaussianBlur(background_gray, (5,5), 0)
    diff = cv.absdiff(blurred_frame, blurred_bg)

    _, diff_mask = cv.threshold(diff, 80, 255, cv.THRESH_BINARY)

    # Combine both masks
    combined_mask = cv.bitwise_or(fg_mask, diff_mask)
    combined_mask = cv.bitwise_or(dark_mask, combined_mask)

    # Morphology operations to remove noise and close gaps
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, (5,5), iterations=2)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, (3,3), iterations=2)
    combined_mask = cv.dilate(combined_mask, (3,3), iterations=2)
    combined_mask = cv.erode(combined_mask, (7,7), iterations=1)


    combined_mask = cv.bitwise_and(combined_mask, combined_mask, mask=roi)

    # Find contours
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rects = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 450:  # Filter small white blobs that are most likely not vehicles
            x, y, w, h = cv.boundingRect(cnt)
            rects.append((x,y,w,h))
    objects = tracker.update(rects)

    for object in objects.keys():
        total_cars.add(object)

    for object_id, (centroid, bbox) in objects.items():
        x, y, w, h = bbox
        cx, cy = centroid
        cv.putText(frame, f"ID {object_id}", (cx - 10, cy - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Write the stats from in the upper left corner of the window
    cv.rectangle(frame, status_corner[0], status_corner[1], (255, 255, 255), -1)
    cv.rectangle(frame, status_corner[0], status_corner[1], (0,0,0), 2)

    sx, sy = (10,30)

    cv.putText(frame, f"Stats:"
                      f"Total cars: {len(total_cars)}", (sx, sy), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    line_spacing = 20
    sy += line_spacing

    cv.putText(frame, "Entries:", (sx, sy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    sy += line_spacing
    for entry_name, count in tracker.entry_counts.items():
        cv.putText(frame, f"  {entry_name}: {count}", (sx, sy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        sy += line_spacing

    cv.putText(frame, "Exits:", (sx, sy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    sy += line_spacing

    for exit_name, count in tracker.exit_counts.items():
        cv.putText(frame, f"   {exit_name} : {count}", (sx, sy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        sy += line_spacing


    # Display
    cv.imshow("Detected Cars", frame)
    cv.imshow("Combined Mask", combined_mask)
    out.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    times_per_frame[cap.get(cv.CAP_PROP_POS_FRAMES)] = time.time() - start


cap.release()
bg_cap.release()
out.release()

counts_output = {
    "entry_counts": tracker.entry_counts,
    "exit_counts": tracker.exit_counts
}


with open("zone_counts.json", "w") as outfile:
    json.dump(counts_output, outfile, indent=4)

with open("times_per_frame.json", "w") as outfile:
    json.dump(times_per_frame, outfile, indent=4)

cv.destroyAllWindows()
