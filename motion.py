"""Short optical-flow bridge for confirmed items missed by the detector.

This supplies packing evidence only: it never creates identities or increments counts.
"""
import cv2
import numpy as np


class MotionBridge:
    def __init__(self):
        self.previous = None
        self.points = {}

    def update(self, frame, objects, second):
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
        for item in objects.values():
            if item['cls'] != 0 or not item['confirmed']:
                continue
            identity = item['id']
            visible = second - item['last'] < .01
            if visible:
                item['visual_box'] = list(item['box'])
                item['visual_last'] = second
                item['visual_source'] = 'detector'
                self.points[identity] = self.features(gray, item['box'])
            elif second - item['last'] <= 8 and self.previous is not None:
                points = self.points.get(identity)
                if points is None or len(points) < 5:
                    continue
                moved, valid, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous, gray, points, None, winSize=(21,21), maxLevel=2)
                if moved is None:
                    self.points[identity] = None
                    continue
                back, back_valid, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous, moved, None, winSize=(21,21), maxLevel=2)
                if back is None:
                    self.points[identity] = None
                    continue
                error = np.linalg.norm(points-back, axis=2).ravel()
                good = valid.ravel().astype(bool) & back_valid.ravel().astype(bool) & (error < 1.0)
                if good.sum() < 5:
                    self.points[identity] = None
                    continue
                displacement = (moved[good]-points[good]).reshape(-1,2)*2
                delta = np.median(displacement, axis=0)
                coherent = np.linalg.norm(displacement-delta, axis=1) < 8
                if coherent.sum() < 5 or np.linalg.norm(delta) > 60:
                    self.points[identity] = None
                    continue
                box = np.array(item.get('visual_box', item['box'])) + np.tile(delta,2)
                if box[0] < 0 or box[1] < 0 or box[2] > frame.shape[1] or box[3] > frame.shape[0]:
                    self.points[identity] = None
                    continue
                item['visual_box'] = box.tolist()
                item['visual_last'] = second
                item['visual_source'] = 'optical_flow'
                item['visual_motion'] = item.get('visual_motion', 0.) + float(np.linalg.norm(delta))
                self.points[identity] = moved[good][coherent]
            else:
                self.points.pop(identity, None)
        self.previous = gray

    @staticmethod
    def features(gray, box):
        x1,y1,x2,y2 = [round(v*.5) for v in box]
        x1,y1 = max(0,x1),max(0,y1)
        x2,y2 = min(gray.shape[1],x2),min(gray.shape[0],y2)
        if x2-x1 < 5 or y2-y1 < 5:
            return None
        points = cv2.goodFeaturesToTrack(gray[y1:y2,x1:x2], maxCorners=30,
                                       qualityLevel=.02,minDistance=3,blockSize=3)
        return None if points is None else points + np.array([[[x1,y1]]],np.float32)
