from collections import deque
import copy


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
        self.CPM_lastIncluded = None  # Last included perception on a CPM
        # Metadata
        self.id = id
        self.detected = detected  # False if this is a CV
        self.onSight = onSight  # False if this is obj is not currently on sight
        self.perceivedBy = []
        self.kalman_filter = None
        self.tracked = False  # True if this object has been tracked for at least 10 frames
        self.CPM = False  # True if this object has been perceived from a CPM

    def insertPerception(self, obj):
        self.perception = obj
        if len(self.pathHistory) >= 10:
            self.pathHistory.popleft()
        self.pathHistory.append(copy.deepcopy(obj))
        if len(self.pathHistory) == 10 or not self.detected or self.CPM:
            self.tracked = True

    def getLatestPoint(self):
        return self.pathHistory[(len(self.pathHistory) - 1)]

    def getOldestPoint(self):
        return self.pathHistory[0]


class Perception:
    def __init__(self, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 o3d_bbx=None, dxSpeed=0, dySpeed=0, fromID=None, ID=None, yaw = 0):
        self.id = ID
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.width = width
        self.length = length
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.speed = 0
        self.heading = heading  # Heading value indicates the angle of speed vector
        self.yaw = yaw  # Yaw value indicates the 'orientation' angle of the object
        self.timestamp = timestamp
        self.confidence = confidence
        self.min = None
        self.max = None
        self.o3d_bbx = o3d_bbx  # Open3D axis alligned bounding box
        self.line_set = None  # Open3D line set
        self.o3d_obb = None  # Open3D oriented bounding box
        self.dxSpeed = dxSpeed
        self.dySpeed = dySpeed
        self.xacc = 0
        self.yacc = 0
        self.fromID = fromID


class PLDMentry:
    def __init__(self, id, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 o3d_bbx=None, detected=True, connected=False, assignedPM=None, onSight=True):
        self.perception = Perception(xPosition, yPosition, width, length, timestamp, confidence,
                                     xSpeed=xSpeed, ySpeed=ySpeed, heading=heading, o3d_bbx=o3d_bbx)
        self.perception.detected = detected
        self.pathHistory = deque([], maxlen=10)
        self.CPM_lastIncluded = None  # Last included perception on a CPM
        # Metadata
        self.id = id
        self.detected = detected
        self.assignedPM = assignedPM
        self.newPO = False
        self.PM = None
        self.onSight = onSight
        self.perceivedBy = []
        self.tracked = False
        self.CPM = False  # True if this object has been perceived from a CPM
        self.kalman_filter = None

    def insertPerception(self, obj):
        self.perception = obj
        if len(self.pathHistory) >= 10:
            self.pathHistory.popleft()
        self.pathHistory.append(copy.deepcopy(obj))
        if len(self.pathHistory) == 10 or not self.detected or self.CPM:
            self.tracked = True

    def getLatestPoint(self):
        return self.pathHistory[(len(self.pathHistory) - 1)]

    def getOldestPoint(self):
        return self.pathHistory[0]


def interpolate_color(color1, color2, ratio):
    return [
        int(color1[i] + (color2[i] - color1[i]) * ratio) for i in range(3)
    ]


class ColorGradient:
    def __init__(self, steps):
        self.steps = steps
        self.gradient = self.generate_gradient()

    def generate_gradient(self):
        # Define the colors for green, yellow, and red
        steps = self.steps
        green = [0, 255, 0]
        yellow = [255, 255, 0]
        red = [255, 0, 0]

        # Calculate the steps for each transition (half the total steps for green-to-yellow and yellow-to-red)
        half_steps = steps // 2

        gradient = []

        # Generate the green-to-yellow portion of the gradient
        for i in range(half_steps):
            ratio = i / half_steps
            color = interpolate_color(green, yellow, ratio)
            gradient.append(color)

        # Generate the yellow-to-red portion of the gradient
        for i in range(half_steps, steps):
            ratio = (i - half_steps) / half_steps
            color = interpolate_color(yellow, red, ratio)
            gradient.append(color)

        return gradient
