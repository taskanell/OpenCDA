import threading

from opencda.core.common.vehicle_manager import VehicleManager
from collections import deque
import math
import numpy as np
import open3d as o3d
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, o3d_visualizer_showLDM
from opencda.customize.v2x.aux import LDMObject
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
import opencda.core.sensing.perception.sensor_transformation as st
import csv
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment
from opencda.customize.v2x.aux import LDMObject
from opencda.customize.v2x.aux import LDMentry
from opencda.customize.v2x.aux import newLDMentry
from opencda.customize.v2x.aux import PLDMentry
from opencda.customize.v2x.aux import Perception


class speed_kalman_filter:
    def __init__(self):
        self.f1 = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = 0.05  # time step
        self.f1.F = np.array([[1, self.dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, self.dt],
                              [0, 0, 0, 1]])
        self.f1.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])

        self.f1.R = np.array([[0.5, 0],
                              [0, 0.5]])
        self.f1.Q = np.eye(8) * 0.2

    def init_step(self, x, y):
        self.f1.x = np.array([[0, 0, 0, 0]]).T
        self.f1.P = np.eye(4) * 500.

    def step(self, x, y):
        z = np.array([[x], [y]])
        self.f1.predict()
        self.f1.update(z)

        # x , y, xSpeed, ySpeed
        return self.f1.x[0, 0], self.f1.x[2, 0], self.f1.x[1, 0], self.f1.x[3, 0]


class PO_kalman_filter:
    def __init__(self):
        self.f1 = KalmanFilter(dim_x=8, dim_z=4)
        self.dt = 0.05  # time step
        self.f1.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, self.dt, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, self.dt, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, self.dt],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        self.f1.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
        # todo: refine this values
        self.f1.R = np.array([[10, 0, 0, 0],
                              [0, 10, 0, 0],
                              [0, 0, 10, 0],
                              [0, 0, 0, 10]])
        self.f1.Q = np.eye(8) * 0.2

    def init_step(self, x, y, xe, ye):
        self.f1.x = np.array([[x, 0, y, 0, xe, 0, ye, 0]]).T
        self.f1.P = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 10, 0, 0, 0],
                              [0, 0, 0, 0, 0, 100, 0, 0],
                              [0, 0, 0, 0, 0, 0, 10, 0],
                              [0, 0, 0, 0, 0, 0, 0, 100]])

    def predict(self):
        self.f1.predict()
        # x , y, extentX, extentY, xSpeed, ySpeed
        return self.f1.x[0, 0], self.f1.x[2, 0], self.f1.x[4, 0], self.f1.x[6, 0], self.f1.x[1, 0], self.f1.x[3, 0]

    def update(self, x, y, xe, ye):
        z = np.array([[x], [y], [xe], [ye]])
        self.f1.update(z)
        # x , y, extentX, extentY, xSpeed, ySpeed
        return self.f1.x[0, 0], self.f1.x[2, 0], self.f1.x[4, 0], self.f1.x[6, 0], self.f1.x[1, 0], self.f1.x[3, 0]

    def step(self, x, y, xe, ye):
        z = np.array([[x], [y], [xe], [ye]])
        self.f1.predict()
        self.f1.update(z)

        # x , y, extentX, extentY, xSpeed, ySpeed
        return self.f1.x[0, 0], self.f1.x[2, 0], self.f1.x[4, 0], self.f1.x[6, 0], self.f1.x[1, 0], self.f1.x[3, 0]


def compute_IoU(box1, box2):
    # Calculate the Intersection over Union between to bbx
    intersection_min = np.maximum(box1.min_bound, box2.min_bound)
    intersection_max = np.minimum(box1.max_bound, box2.max_bound)
    intersection_size = np.maximum(0, intersection_max - intersection_min)

    volume_box1 = np.prod(box1.get_extent())
    volume_box2 = np.prod(box2.get_extent())

    volume_intersection = np.prod(intersection_size)
    volume_union = volume_box1 + volume_box2 - volume_intersection
    if volume_union == 0:
        iou = 0
    else:
        iou = volume_intersection / volume_union

    return iou


def CAMfusion(cav, CAMobject):
    CAMobject.connected = True
    CAMobject.o3d_bbx = get_o3d_bbx(cav,
                                    CAMobject.xPosition,
                                    CAMobject.yPosition,
                                    CAMobject.width,
                                    CAMobject.length)
    if CAMobject.id in cav.LDM:
        # If this is not the first CAM
        cav.LDM[CAMobject.id].insertPerception(CAMobject)
    else:
        # If this is the first CAM, check if we are already perceiving it
        IoU_map, new, matched, ldm_ids = match_LDM(cav, [CAMobject])
        if IoU_map is not None:
            if IoU_map[matched[0], new[0]] >= 0:
                # If we are perceiving this object, delete the entry as PO
                del cav.LDM[ldm_ids[matched[0]]]
                cav.LDM_ids.add(ldm_ids[matched[0]])
        # Create new entry
        if CAMobject.id in cav.LDM_ids:
            cav.LDM_ids.remove(CAMobject.id)
        cav.LDM[CAMobject.id] = newLDMentry(CAMobject, CAMobject.id, detected=False, onSight=True)
        cav.LDM[CAMobject.id].kalman_filter = PO_kalman_filter()
        cav.LDM[CAMobject.id].kalman_filter.init_step(CAMobject.xPosition,
                                                      CAMobject.yPosition,
                                                      CAMobject.width,
                                                      CAMobject.length)
    return CAMobject.id


def CPMfusion(cav, object_list, fromID):
    # Try to match CPM objects with LDM ones
    # If we match an object, we perform fusion averaging the bbx
    # If can't match the object we append it to the LDM as a new object

    for CPMobj in object_list:
        if cav.time > CPMobj.timestamp:
            # If it's an old perception, we need to predict its current position
            CPMobj.xPosition += CPMobj.xSpeed * (cav.time - CPMobj.timestamp)
            CPMobj.yPosition += CPMobj.ySpeed * (cav.time - CPMobj.timestamp)
            CPMobj.o3d_bbx = get_o3d_bbx(cav, CPMobj.xPosition,
                                         CPMobj.yPosition,
                                         CPMobj.width,
                                         CPMobj.length)

    IoU_map, new, matched, ldm_ids = match_LDM(cav, object_list)

    for j in range(len(object_list)):
        CPMobj = object_list[j]
        # Compute bbx from cav's POV because we already converted values
        CPMobj.o3d_bbx = get_o3d_bbx(cav, CPMobj.xPosition,
                                     CPMobj.yPosition,
                                     CPMobj.width,
                                     CPMobj.length)
        if IoU_map is not None:
            matchedObj = matched[np.where(new == j)[0]]
            if IoU_map[matchedObj, j] >= 0:
                append_CPM_object(cav, CPMobj, ldm_ids[matchedObj[0]], fromID)
                continue
        newID = cav.LDM_ids.pop()
        cav.LDM[newID] = newLDMentry(CPMobj, newID, detected=True, onSight=False)
        cav.LDM[newID].kalman_filter = PO_kalman_filter()
        cav.LDM[newID].kalman_filter.init_step(CPMobj.xPosition, CPMobj.yPosition, CPMobj.width, CPMobj.length)


def append_CPM_object(cav, CPMobj, id, fromID):
    if fromID not in cav.LDM[id].perceivedBy:
        cav.LDM[id].perceivedBy.append(fromID)
    if CPMobj.timestamp < cav.LDM[id].getLatestPoint().timestamp - 100:  # Consider objects up to 100ms old
        return

    newLDMobj = Perception(CPMobj.xPosition,
                           CPMobj.yPosition,
                           CPMobj.width,
                           CPMobj.length,
                           CPMobj.timestamp,
                           CPMobj.confidence)
    # If the object is also perceived locally
    if cav.vehicle.id in cav.LDM[id].perceivedBy:
        # Compute weights depending on the POage and confidence (~distance from detecting vehicle)
        LDMobj = cav.LDM[id].perception
        LDMobj_age = LDMobj.timestamp - cav.time
        CPMobj_age = CPMobj.timestamp - cav.time
        if (LDMobj.confidence == 0 and CPMobj.confidence == 0) or \
                (LDMobj_age == 0 and CPMobj_age == 0):
            weightCPM = 0.5
            weightLDM = 0.5
        else:
            weightLDM = (LDMobj_age / (CPMobj_age + LDMobj_age)) * \
                        (LDMobj.confidence / (LDMobj.confidence + CPMobj.confidence))
            weightCPM = (CPMobj_age / (CPMobj_age + LDMobj_age)) * \
                        (CPMobj.confidence / (LDMobj.confidence + CPMobj.confidence))
            if (weightLDM + weightCPM) == 0:
                weightLDM = 0.5
                weightCPM = 0.5

        newLDMobj.xPosition = (LDMobj.xPosition * weightCPM + CPMobj.xPosition * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.yPosition = (LDMobj.yPosition * weightCPM + CPMobj.yPosition * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.xSpeed = (CPMobj.xSpeed * weightCPM + LDMobj.xSpeed * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.ySpeed = (CPMobj.ySpeed * weightCPM + LDMobj.ySpeed * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.width = (CPMobj.width * weightCPM + LDMobj.width * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.length = (CPMobj.length * weightCPM + LDMobj.length * weightLDM) / (weightLDM + weightCPM)
        newLDMobj.heading = (CPMobj.heading * weightCPM + LDMobj.heading * weightLDM) / (weightLDM + weightCPM)
        # newLDMobj.confidence = (CPMobj.confidence + LDMobj.confidence) / 2
        newLDMobj.timestamp = cav.time
        cav.LDM[id].onSight = True
    else:
        cav.LDM[id].onSight = False

    newLDMobj.o3d_bbx = get_o3d_bbx(cav,
                                    newLDMobj.xPosition,
                                    newLDMobj.yPosition,
                                    newLDMobj.width,
                                    newLDMobj.length)
    # cav.LDM[id].kalman_filter.update(newLDMobj.xPosition, newLDMobj.yPosition, newLDMobj.width, newLDMobj.length)
    cav.LDM[id].insertPerception(newLDMobj)


def match_LDM(cav, object_list):
    if len(cav.LDM) != 0:
        IoU_map = np.zeros((len(cav.LDM), len(object_list)), dtype=np.float32)
        i = 0
        ldm_ids = []
        for ID, LDMobj in cav.LDM.items():
            for j in range(len(object_list)):
                obj = object_list[j]
                object_list[j].o3d_bbx = cav.LDMobj_to_o3d_bbx(obj)
                LDMpredX = LDMobj.perception.xPosition
                LDMpredY = LDMobj.perception.yPosition
                LDMpredbbx = LDMobj.perception.o3d_bbx
                if cav.time > LDMobj.perception.timestamp:
                    LDMpredX += (cav.time - LDMobj.perception.timestamp) * LDMobj.perception.xSpeed
                    LDMpredY += (cav.time - LDMobj.perception.timestamp) * LDMobj.perception.ySpeed
                    LDMpredbbx = cav.LDMobj_to_o3d_bbx(LDMobj.perception)
                LDMpredbbx = get_o3d_bbx(cav, LDMpredX, LDMpredY, LDMobj.perception.width, LDMobj.perception.length)

                dist = math.sqrt(
                    math.pow((obj.xPosition - LDMpredX), 2) + math.pow((obj.yPosition - LDMpredY), 2))
                iou = compute_IoU(LDMpredbbx, object_list[j].o3d_bbx)
                if iou > 0:
                    IoU_map[i, j] = iou
                elif dist > 3:  # if dist < 3 --> IoU_map[i, j] = 0
                    IoU_map[i, j] = -1000
            i += 1
            ldm_ids.append(ID)
        matched, new = linear_assignment(-IoU_map)
        return IoU_map, new, matched, ldm_ids
    else:
        return None, None, None, None


def LDM2OpencdaObj(cav, LDM, trafficLights):
    retObjects = []
    for ID, LDMObject in LDM.items():
        corner = np.asarray(LDMObject.perception.o3d_bbx.get_box_points())
        # covert back to unreal coordinate
        corner[:, :1] = -corner[:, :1]
        corner = corner.transpose()
        # extend (3, 8) to (4, 8) for homogenous transformation
        corner = np.r_[corner, [np.ones(corner.shape[1])]]
        # project to world reference
        corner = st.sensor_to_world(corner, cav.perception_manager.lidar.sensor.get_transform())
        corner = corner.transpose()[:, :3]
        object = ObstacleVehicle(corner, LDMObject.perception.o3d_bbx)
        object.carla_id = LDMObject.id
        retObjects.append(object)

    return {'vehicles': retObjects, 'traffic_lights': trafficLights}


def LDM_to_lidarObjects(PLDM):
    lidarObjects = []
    for ID, LDMobj in PLDM.PLDM.items():
        lidarObjects.append(LDMobj)  # return last sample of each object in LDM
    return {'vehicles': lidarObjects}


def matchLDMobject(PLDM, object):
    matched = False
    matchedId = -1
    for ID, LDMobj in PLDM.items():
        LDMcurrX = LDMobj.xPosition
        LDMcurrY = LDMobj.yPosition
        if PLDM.cav.time > LDMobj.timestamp:
            LDMcurrX += (PLDM.cav.time - LDMobj.timestamp) * LDMobj.xSpeed
            LDMcurrY += (PLDM.cav.time - LDMobj.timestamp) * LDMobj.ySpeed
        if abs(object.xPosition - LDMcurrX) < 3 and abs(object.yPosition - LDMcurrY) < 3 and LDMobj.id != object.id:
            # matched = True
            matchedId = ID
            break
    return matchedId


def LDMobj_to_o3d_bbx(cav, LDMobj):
    # o3d bbx test
    lidarPos = cav.perception_manager.lidar.sensor.get_transform()
    objRelPos = np.array([-1 * (LDMobj.xPosition - lidarPos.location.x),
                          LDMobj.yPosition - lidarPos.location.y,
                          1.5 - lidarPos.location.z])
    min_arr = objRelPos - np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    max_arr = objRelPos + np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    # Reshape the array to have a shape of (3, 1)
    min_arr = min_arr.reshape((3, 1))
    max_arr = max_arr.reshape((3, 1))
    # Assign the array to the variable min_bound
    min_bound = min_arr.astype(np.float64)
    max_bound = max_arr.astype(np.float64)
    geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)
    return geometry


def get_o3d_bbx(cav, x, y, xe, ye):
    # o3d bbx test
    lidarPos = cav.perception_manager.lidar.sensor.get_transform()
    objRelPos = np.array([-1 * (x - lidarPos.location.x),
                          y - lidarPos.location.y,
                          1.5 - lidarPos.location.z])
    min_arr = objRelPos - np.array([xe / 2, ye / 2, 0.75])
    max_arr = objRelPos + np.array([xe / 2, ye / 2, 0.75])
    # Reshape the array to have a shape of (3, 1)
    min_arr = min_arr.reshape((3, 1))
    max_arr = max_arr.reshape((3, 1))
    # Assign the array to the variable min_bound
    min_bound = min_arr.astype(np.float64)
    max_bound = max_arr.astype(np.float64)
    geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)
    return geometry


def obj_to_o3d_bbx(PLDM, obj):
    lidarPos = PLDM.cav.perception_manager.lidar.sensor.get_transform()
    objRelPos = np.array([-1 * (obj.bounding_box.location.x - lidarPos.location.x),
                          obj.bounding_box.location.y - lidarPos.location.y,
                          1.5 - lidarPos.location.z])
    min_arr = objRelPos - np.array([obj.bounding_box.extent.x, obj.bounding_box.extent.y, 0.75])
    max_arr = objRelPos + np.array([obj.bounding_box.extent.x, obj.bounding_box.extent.y, 0.75])
    # Reshape the array to have a shape of (3, 1)
    min_arr = min_arr.reshape((3, 1))
    max_arr = max_arr.reshape((3, 1))
    # Assign the array to the variable min_bound
    min_bound = min_arr.astype(np.float64)
    max_bound = max_arr.astype(np.float64)
    geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)
    return geometry


def LDMobj_to_min_max(PLDM, LDMobj):
    # o3d bbx test
    lidarPos = PLDM.cav.perception_manager.lidar.sensor.get_transform()
    objRelPos = np.array([-1 * (LDMobj.xPosition - lidarPos.location.x),
                          LDMobj.yPosition - lidarPos.location.y,
                          1.5 - lidarPos.location.z])
    min_arr = objRelPos - np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    max_arr = objRelPos + np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    return min_arr, max_arr
