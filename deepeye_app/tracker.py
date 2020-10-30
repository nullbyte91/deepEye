from collections import OrderedDict

import cv2
from scipy.spatial import distance as dist

import numpy as np


class Tracker:
    def __init__(self, log, maxDisappeared=50, maxDistance=50, maxHistory=100, ):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.maxHistory = maxHistory
        self.obj_class = OrderedDict()
        self.colors = OrderedDict()
        self.history = OrderedDict()

        self.log = log

    def register(self, centroid, obj_cls):
        self.history[self.nextObjectID] = [centroid]
        self.objects[self.nextObjectID] = centroid
        self.obj_class[self.nextObjectID] = obj_cls
        self.disappeared[self.nextObjectID] = 0
        self.colors[self.nextObjectID] = np.random.choice(range(256), size=3).astype('i').tolist()
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.obj_class[objectID]
        del self.colors[objectID]
        del self.history[objectID]

    def update(self, pts_l, log):
        pts = []
        for item in pts_l:
            pts.append(item[0:2])
        
        if len(pts) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects, self.obj_class

        if len(self.objects) == 0:
            for pt in pts_l:
                self.register(pt[0:2], pt[2:3])

        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), pts)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = pts[col]
                self.history[objectID] = (self.history[objectID] + [pts[col]])[-self.maxHistory:]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(pts[col], "None")

        return self.objects, self.obj_class
