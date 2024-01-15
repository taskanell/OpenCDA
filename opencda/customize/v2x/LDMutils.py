import threading

from opencda.core.common.vehicle_manager import VehicleManager
from collections import deque
import math
import numpy as np
import open3d as o3d
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, o3d_visualizer_showLDM
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
import opencda.core.sensing.perception.sensor_transformation as st
import csv
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment as linear_assignment
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
    def __init__(self, dt=0.05):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        self.dt = dt  # time step
        self.last_predict = 0
        self.last_update = 0
        self.kf.F = np.array([[1, 0, self.dt, 0, 0.5 * self.dt ** 2, 0],
                              [0, 1, 0, self.dt, 0, 0.5 * self.dt ** 2],
                              [0, 0, 1, 0, self.dt, 0],
                              [0, 0, 0, 1, 0, self.dt],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0]])
        # todo: refine this values
        self.kf.R = np.array([[10, 0],
                              [0, 10]])
        noise_ax = 1
        noise_ay = 1
        self.kf.Q = np.array(
            [[self.dt ** 4 / 4 * noise_ax, 0, self.dt ** 3 / 2 * noise_ax, 0, self.dt ** 2 * noise_ax, 0],
             [0, self.dt ** 4 / 4 * noise_ay, 0, self.dt ** 3 / 2 * noise_ay, 0, self.dt ** 2 * noise_ay],
             [self.dt ** 3 / 2 * noise_ax, 0, self.dt ** 2 * noise_ax, 0, self.dt * noise_ax, 0],
             [0, self.dt ** 3 / 2 * noise_ay, 0, self.dt ** 2 * noise_ay, 0, self.dt * noise_ay],
             [self.dt ** 2 * noise_ax, 0, self.dt * noise_ax, 0, 1, 0],
             [0, self.dt ** 2 * noise_ay, 0, self.dt * noise_ay, 0, 1]])

    def init_step(self, x, y, vx=0, vy=0, ax=0, ay=0):
        self.kf.P *= 1e-4
        # Initial state [x, y, vx, vy, ax, ay]
        self.kf.x = np.array([[x], [y], [vx], [vy], [ax], [ay]])

    def predict(self, now=None):
        if now is not None:
            if now - self.last_predict >= self.dt:
                self.kf.predict()
                # print('Predict: ', now, ' - dt: ', now - self.last_predict)
        self.last_predict = now
        # [x, y, vx, vy, ax, ay]
        return self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0], self.kf.x[3, 0], self.kf.x[4, 0], self.kf.x[5, 0]

    def update(self, x, y, now=None):
        if now is not None:
            if now - self.last_update >= self.dt:
                z = np.array([[x], [y]])
                self.kf.update(z)
                # print('Update: ', now, ' - dt: ', now - self.last_update)
        # z = np.array([[x], [y]])
        # self.kf.update(z)
        self.last_update = now
        # x , y, extentX, extentY, xSpeed, ySpeed
        return self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0], self.kf.x[3, 0], self.kf.x[4, 0], self.kf.x[5, 0]


def compute_IoU(box1, box2):
    # Calculate the Intersection over Union between two bbx
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


def compute_IoU_lineSet(set1, set2):
    # Calculate the Intersection over Union between two bbx
    intersection_min = np.maximum(set1.get_min_bound(), set2.get_min_bound())
    intersection_max = np.minimum(set1.get_max_bound(), set2.get_max_bound())
    intersection_size = np.maximum(0, intersection_max - intersection_min)

    points1 = np.asarray(set1.points)
    points2 = np.asarray(set2.points)
    length1 = np.linalg.norm(points1[0] - points1[1])
    length2 = np.linalg.norm(points2[0] - points2[1])
    width1 = np.linalg.norm(points1[1] - points1[2])
    width2 = np.linalg.norm(points2[1] - points2[2])
    height1 = np.linalg.norm(points1[2] - points1[4])
    height2 = np.linalg.norm(points2[2] - points2[4])

    volume_box1 = length1 * width1 * height1
    volume_box2 = length2 * width2 * height2
    # volume_box1 = np.prod(set1.get_oriented_bounding_box().extent)
    # volume_box2 = np.prod(set2.get_oriented_bounding_box().extent)

    volume_intersection = np.prod(intersection_size)
    volume_union = volume_box1 + volume_box2 - volume_intersection
    if volume_union == 0:
        iou = 0
    else:
        iou = volume_intersection / volume_union

    return iou


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
    lidarPos = cav.perception_manager.lidar.sensor.get_transform()
    translation = [LDMobj.xPosition, LDMobj.yPosition, lidarPos.location.z + 1.5]
    h, l, w = 1.5, LDMobj.length, LDMobj.width
    rotation = np.deg2rad(LDMobj.yaw)

    # Create a bounding box outline
    bounding_box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    new_element = np.array([[1, 1, 1, 1, 1, 1, 1, 1]]) # Needed for world_to_sensor method to work
    corner_box = np.append(corner_box, new_element, axis=0)

    corner_box_sensor = st.world_to_sensor(corner_box, lidarPos)

    corner_box_sensor[:1, :] = - corner_box_sensor[:1, :]  # Take the 1s out of the array
    corner_box_sensor = corner_box_sensor[:-1, :]   # convert unreal space to o3d space
    corner_box_sensor = corner_box_sensor.transpose()

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box_sensor)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # TODO take this part out and rely only on line_sets for lidar visualization
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    local_corners = np.array([
        [-LDMobj.width / 2, -LDMobj.length / 2],
        [LDMobj.width / 2, -LDMobj.length / 2],
        [LDMobj.width / 2, LDMobj.length / 2],
        [-LDMobj.width / 2, LDMobj.length / 2]
    ])
    world_corners = np.dot(R, local_corners.T).T + np.array([LDMobj.xPosition, LDMobj.yPosition])

    min_boundary = np.array([np.min(world_corners, axis=0)[0],
                             np.min(world_corners, axis=0)[1],
                             lidarPos.location.z,
                             1])
    max_boundary = np.array([np.max(world_corners, axis=0)[0],
                             np.max(world_corners, axis=0)[1],
                             lidarPos.location.z + 1.5,
                             1])
    min_boundary = min_boundary.reshape((4, 1))
    max_boundary = max_boundary.reshape((4, 1))
    stack_boundary = np.hstack((min_boundary, max_boundary))
    # the boundary coord at the lidar sensor space
    stack_boundary_sensor_cords = st.world_to_sensor(stack_boundary,
                                                     lidarPos)
    # convert unreal space to o3d space
    stack_boundary_sensor_cords[:1, :] = - \
        stack_boundary_sensor_cords[:1, :]
    # (4,2) -> (3, 2)
    stack_boundary_sensor_cords = stack_boundary_sensor_cords[:-1, :]

    min_boundary_sensor = np.min(stack_boundary_sensor_cords, axis=1)
    max_boundary_sensor = np.max(stack_boundary_sensor_cords, axis=1)

    geometry = o3d.geometry.AxisAlignedBoundingBox(min_boundary_sensor, max_boundary_sensor)
    # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)

    return geometry, line_set
    # lidarPos = cav.perception_manager.lidar.sensor.get_transform()
    # objRelPos = np.array([-1 * (LDMobj.xPosition - lidarPos.location.x),
    #                       LDMobj.yPosition - lidarPos.location.y,
    #                       1.5 - lidarPos.location.z])
    # min_arr = objRelPos - np.array([LDMobj.width / 2, LDMobj.length / 2, 0])
    # max_arr = objRelPos + np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    # # Reshape the array to have a shape of (3, 1)
    # min_arr = min_arr.reshape((3, 1))
    # max_arr = max_arr.reshape((3, 1))
    # # Assign the array to the variable min_bound
    # min_bound = min_arr.astype(np.float64)
    # max_bound = max_arr.astype(np.float64)
    # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # geometry.color = (0, 1, 0)
    # return geometry


# def get_o3d_bbx(cav, x, y, xe, ye):
#     # o3d bbx test
#     lidarPos = cav.perception_manager.lidar.sensor.get_transform()
#     objRelPos = np.array([-1 * (x - lidarPos.location.x),
#                           y - lidarPos.location.y,
#                           1.5 - lidarPos.location.z])
#     min_arr = objRelPos - np.array([xe / 2, ye / 2, 0.75])
#     max_arr = objRelPos + np.array([xe / 2, ye / 2, 0.75])
#     # Reshape the array to have a shape of (3, 1)
#     min_arr = min_arr.reshape((3, 1))
#     max_arr = max_arr.reshape((3, 1))
#     # Assign the array to the variable min_bound
#     min_bound = min_arr.astype(np.float64)
#     max_bound = max_arr.astype(np.float64)
#     geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
#     geometry.color = (0, 1, 0)
#     return geometry

def get_o3d_bbx(cav, x, y, xe, ye, heading):
    lidarPos = cav.perception_manager.lidar.sensor.get_transform()

    translation = [x, y, lidarPos.location.z + 1.5]
    h, l, w = 1.5, ye, xe
    rotation = np.deg2rad(heading)

    # Create a bounding box outline
    bounding_box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    new_element = np.array([[1, 1, 1, 1, 1, 1, 1, 1]])
    corner_box = np.append(corner_box, new_element, axis=0)

    corner_box_sensor = st.world_to_sensor(corner_box, lidarPos)

    corner_box_sensor[:1, :] = - corner_box_sensor[:1, :]
    corner_box_sensor = corner_box_sensor[:-1, :]
    corner_box_sensor = corner_box_sensor.transpose()

    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box_sensor)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    yaw = np.deg2rad(heading)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local_corners = np.array([
        [-xe / 2, -ye / 2],
        [xe / 2, -ye / 2],
        [xe / 2, ye / 2],
        [-xe / 2, ye / 2]
    ])
    world_corners = np.dot(R, local_corners.T).T + np.array([x, y])

    min_boundary = np.array([np.min(world_corners, axis=0)[0],
                             np.min(world_corners, axis=0)[1],
                             lidarPos.location.z,
                             1])
    max_boundary = np.array([np.max(world_corners, axis=0)[0],
                             np.max(world_corners, axis=0)[1],
                             lidarPos.location.z + 1.5,
                             1])
    min_boundary = min_boundary.reshape((4, 1))
    max_boundary = max_boundary.reshape((4, 1))
    stack_boundary = np.hstack((min_boundary, max_boundary))
    # the boundary coord at the lidar sensor space
    stack_boundary_sensor_cords = st.world_to_sensor(stack_boundary,
                                                     lidarPos)
    # convert unreal space to o3d space
    stack_boundary_sensor_cords[:1, :] = - \
        stack_boundary_sensor_cords[:1, :]
    # (4,2) -> (3, 2)
    stack_boundary_sensor_cords = stack_boundary_sensor_cords[:-1, :]

    min_boundary_sensor = np.min(stack_boundary_sensor_cords, axis=1)
    max_boundary_sensor = np.max(stack_boundary_sensor_cords, axis=1)

    geometry = o3d.geometry.AxisAlignedBoundingBox(min_boundary_sensor, max_boundary_sensor)
    # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)
    return geometry, line_set


def obj_to_o3d_bbx(PLDM, obj):
    lidarPos = PLDM.cav.perception_manager.lidar.sensor.get_transform()

    yaw = np.deg2rad(obj.yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local_corners = np.array([
        [-obj.width / 2, -obj.length / 2],
        [obj.width / 2, -obj.length / 2],
        [obj.width / 2, obj.length / 2],
        [-obj.width / 2, obj.length / 2]
    ])
    world_corners = np.dot(R, local_corners.T).T + np.array([obj.xPosition, obj.yPosition])

    min_boundary = np.array([np.min(world_corners, axis=0)[0],
                             np.min(world_corners, axis=0)[1],
                             lidarPos.location.z,
                             1])
    max_boundary = np.array([np.max(world_corners, axis=0)[0],
                             np.max(world_corners, axis=0)[1],
                             lidarPos.location.z + 1.5,
                             1])
    min_boundary = min_boundary.reshape((4, 1))
    max_boundary = max_boundary.reshape((4, 1))
    stack_boundary = np.hstack((min_boundary, max_boundary))
    # the boundary coord at the lidar sensor space
    stack_boundary_sensor_cords = st.world_to_sensor(stack_boundary,
                                                     lidarPos)
    # convert unreal space to o3d space
    stack_boundary_sensor_cords[:1, :] = - \
        stack_boundary_sensor_cords[:1, :]
    # (4,2) -> (3, 2)
    stack_boundary_sensor_cords = stack_boundary_sensor_cords[:-1, :]

    min_boundary_sensor = np.min(stack_boundary_sensor_cords, axis=1)
    max_boundary_sensor = np.max(stack_boundary_sensor_cords, axis=1)

    geometry = o3d.geometry.AxisAlignedBoundingBox(min_boundary_sensor, max_boundary_sensor)
    # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    geometry.color = (0, 1, 0)
    return geometry
    # lidarPos = PLDM.cav.perception_manager.lidar.sensor.get_transform()
    # objRelPos = np.array([-1 * (obj.bounding_box.location.x - lidarPos.location.x),
    #                       obj.bounding_box.location.y - lidarPos.location.y,
    #                       1.5 - lidarPos.location.z])
    # min_arr = objRelPos - np.array([obj.bounding_box.extent.x, obj.bounding_box.extent.y, 0.75])
    # max_arr = objRelPos + np.array([obj.bounding_box.extent.x, obj.bounding_box.extent.y, 0.75])
    # # Reshape the array to have a shape of (3, 1)
    # min_arr = min_arr.reshape((3, 1))
    # max_arr = max_arr.reshape((3, 1))
    # # Assign the array to the variable min_bound
    # min_bound = min_arr.astype(np.float64)
    # max_bound = max_arr.astype(np.float64)
    # geometry = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # geometry.color = (0, 1, 0)
    # return geometry


def LDMobj_to_min_max(PLDM, LDMobj):
    # o3d bbx test
    lidarPos = PLDM.cav.perception_manager.lidar.sensor.get_transform()
    objRelPos = np.array([-1 * (LDMobj.xPosition - lidarPos.location.x),
                          LDMobj.yPosition - lidarPos.location.y,
                          1.5 - lidarPos.location.z])
    min_arr = objRelPos - np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    max_arr = objRelPos + np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
    return min_arr, max_arr
