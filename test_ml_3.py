import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import json

from classes.constants import ENTRY_ZONES, EXIT_ZONES

cap = cv.VideoCapture('Inputs/stabilization_with_videostab_4.avi')

model = YOLO('yolov8_best.pt')
# Code made for the video writer
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter.fourcc(*'mp4v')

out = cv.VideoWriter('Outputs/detection_using_personally_trained_model_with_stats_1.mp4',
                     fourcc, fps, (width, height))
entry_counts = {name: 0 for name in ENTRY_ZONES}
exit_counts = {name: 0 for name in EXIT_ZONES}
counted_entry_ids = set()
counted_exit_ids = set()

def point_intersection_with_poly(point, poly):
    return cv.pointPolygonTest(poly, point, False) >= 0

def point_cross_line(point, line, tol=10):
    (x1, y1), (x2, y2) = line
    line_vector = np.array([x2 - x1, y2 - y1])
    point_vector = np.array([point[0] - x1, point[1] - y1])
    proj = np.dot(point_vector, line_vector) / np.linalg.norm(line_vector)

    if proj < 0 or proj > np.linalg.norm(line_vector):
        return False

    dist = np.abs(np.cross(line_vector, point_vector)) / np.linalg.norm(line_vector)
    return dist < tol


time_per_frame = {}

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, stream=True, persist=True)

    for r in results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes
        ids = boxes.id.cpu().tolist()
        clss = boxes.cls.cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()


        for box, obj_id, cls in zip(xyxy, ids, clss):

            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]


            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(frame, f"{label} ID {obj_id}", (x1, y1-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Detection for entries in intersection
            if obj_id not in counted_entry_ids:
                for name, poly in ENTRY_ZONES.items():
                    if point_intersection_with_poly(center, poly):
                        entry_counts[name] += 1
                        counted_entry_ids.add(obj_id)
                        break
            if obj_id not in counted_exit_ids:
                for name, line in EXIT_ZONES.items():
                    if point_cross_line(center, line):
                        exit_counts[name] += 1
                        counted_exit_ids.add(obj_id)
                        break



    cv.imshow("Tracked vehicle", frame)
    out.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    time_per_frame[cap.get(cv.CAP_PROP_POS_FRAMES)] = time.time() - start

cap.release()
out.release()
cv.destroyAllWindows()
output_data = {
    "entry_counts": entry_counts,
    "exit_counts": exit_counts,
}

with open("ml_count.json", "w") as file:
    json.dump(output_data, file, indent=4)
with open("ml_times_per_frame.json", "w") as file:
    json.dump(time_per_frame, file, indent=4)
