from collections import deque
import copy


class LDMObject:
    def __init__(self, id, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 detected=True, o3d_bbx=None, connected=False):
        self.id = id
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.width = width
        self.length = length
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.heading = heading
        self.timestamp = timestamp
        self.confidence = confidence
        self.o3d_bbx = o3d_bbx
        self.min = None
        self.max = None
        self.detected = detected
        self.detectedBy = None
        self.connected = connected
        self.estimated_x = 0
        self.estimated_y = 0
        self.estimated_t = 0
        self.CPM_id = None


def newLDMentry(perception, id, detected=True, onSight=True):
    # Function to create a new LDMentry from a perception
    retEntry = LDMentry(id, perception.xPosition, perception.yPosition, perception.width, perception.length,
                        perception.timestamp, perception.confidence, xSpeed=perception.xSpeed, ySpeed=perception.ySpeed,
                        heading=perception.heading, detected=detected, o3d_bbx=perception.o3d_bbx, onSight=True)
    retEntry.insertPerception(retEntry.perception)
    return retEntry


def newPLDMentry(perception, id, detected=True, onSight=True):
    # Function to create a new LDMentry from a perception
    retEntry = PLDMentry(id, perception.xPosition, perception.yPosition, perception.width, perception.length,
                         perception.timestamp, perception.confidence, xSpeed=perception.xSpeed,
                         ySpeed=perception.ySpeed,
                         heading=perception.heading, detected=detected, o3d_bbx=perception.o3d_bbx, onSight=True)
    retEntry.insertPerception(retEntry.perception)
    return retEntry


class LDMentry:
    def __init__(self, id, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 detected=True, o3d_bbx=None, onSight=True):
        self.perception = Perception(xPosition, yPosition, width, length, timestamp, confidence,
                                     xSpeed=xSpeed, ySpeed=ySpeed, heading=heading, o3d_bbx=o3d_bbx)
        self.perception.detected = detected
        self.pathHistory = deque([], maxlen=10)
        # Metadata
        self.id = id
        self.detected = detected
        self.onSight = onSight
        self.perceivedBy = []
        self.kalman_filter = None

    def insertPerception(self, obj):
        self.perception = obj
        if len(self.pathHistory) >= 10:
            self.pathHistory.popleft()
        self.pathHistory.append(copy.deepcopy(obj))

    def getLatestPoint(self):
        return self.pathHistory[(len(self.pathHistory) - 1)]

    def getOldestPoint(self):
        return self.pathHistory[0]


class Perception:
    def __init__(self, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 o3d_bbx=None, dxSpeed=0, dySpeed=0):
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.width = width
        self.length = length
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.speed = 0
        self.heading = heading
        self.timestamp = timestamp
        self.confidence = confidence
        self.min = None
        self.max = None
        self.o3d_bbx = o3d_bbx
        self.dxSpeed = dxSpeed
        self.dySpeed = dySpeed


class PLDMentry:
    def __init__(self, id, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 o3d_bbx=None, detected=True, connected=False, assignedPM=None, onSight=True):
        self.perception = Perception(xPosition, yPosition, width, length, timestamp, confidence,
                                     xSpeed=xSpeed, ySpeed=ySpeed, heading=heading, o3d_bbx=o3d_bbx)
        self.pathHistory = deque([], maxlen=10)
        # Metadata
        self.id = id
        self.detected = detected
        self.assignedPM = assignedPM
        self.newPO = False
        self.PM = None
        self.onSight = onSight
        self.perceivedBy = []

    def insertPerception(self, obj):
        self.perception = obj
        if len(self.pathHistory) >= 10:
            self.pathHistory.popleft()
        self.pathHistory.append(copy.deepcopy(obj))

    def getLatestPoint(self):
        return self.pathHistory[(len(self.pathHistory) - 1)]

    def getOldestPoint(self):
        return self.pathHistory[0]
