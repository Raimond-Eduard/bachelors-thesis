import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
from classes.constants import EXIT_ZONES, ENTRY_ZONES
import cv2 as cv
import math

class EnhancedCentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=100):
        """

        :param max_disappeared: Maximum number of frames it takes for the class to update
        the position of the centroid
        :param max_distance: The distance for which the tracker decides whether to maintain
        or change the given id to a centroid
        """
        self.next_object_id = 0
        self.objects = OrderedDict()             # object_id -> (centroid, bbox)
        self.disappeared = OrderedDict()         # object_id -> frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # For counting in entries and exits
        self.entry_flags = {} # object_id -> zone_name
        self.previous_centroids = {} # object_id -> previous centroid
        self.entry_counts = {zone: 0 for zone in ENTRY_ZONES}
        self.exit_counts = {line: 0 for line in EXIT_ZONES}


    def register(self, centroid, bbox):
        """

        :param centroid: The centroid of the object
        :param bbox: The bounding box of the object
        :return: void
        """
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.previous_centroids[self.next_object_id] = centroid

        self.next_object_id += 1


    def deregister(self, object_id):
        """
        This function removes the centroid from the object
        :param object_id: The id for the centroid to be removed
        :return:
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.entry_flags:
            del self.entry_flags[object_id]
        if object_id in self.previous_centroids:
            del self.previous_centroids[object_id]


    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_bboxes = []
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids[i] = (cx, cy)
            input_bboxes.append((x, y, w, h))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_data = list(self.objects.values())
            object_centroids = [data[0] for data in object_data]

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row][col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                old_centroid = self.objects[object_id][0]
                new_centroid = input_centroids[col]

                self.previous_centroids[object_id] = old_centroid
                self.objects[object_id] = (new_centroid, input_bboxes[col])
                self.disappeared[object_id] = 0

                # Entry zone detections
                if object_id not in self.entry_flags:
                    for zone_name, polygon in ENTRY_ZONES.items():
                        if cv.pointPolygonTest(polygon, (int(old_centroid[0]), int(old_centroid[1])), False) >= 0:
                            self.entry_flags[object_id] = zone_name
                            self.entry_counts[zone_name] += 1
                            break

                # Exit point line crossing
                if object_id in self.previous_centroids:
                    p_old = self.previous_centroids[object_id]
                    p_new = new_centroid

                    for line_name, (pt1, pt2) in EXIT_ZONES.items():
                        if self._line_crossed(p_old, p_new, pt1, pt2):
                            self.exit_counts[line_name] += 1

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])

        return self.objects

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _line_crossed(self, p1, p2, l1, l2):
        return self._ccw(p1, l1, l2) != self._ccw(p2, l1, l2) and self._ccw(p1, p2, l1) != self._ccw(p1, p2, l2)
