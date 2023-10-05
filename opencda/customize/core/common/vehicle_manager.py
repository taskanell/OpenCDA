import threading

from opencda.core.common.vehicle_manager import VehicleManager
from scipy.optimize import linear_sum_assignment as linear_assignment
# from sklearn.utils.linear_assignment_ import linear_assignment

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
import weakref

from opencda.customize.v2x.v2x_agent import V2XAgent
from opencda.customize.v2x.PLDM import PLDM
from opencda.customize.v2x.LDM import LDM
from opencda.customize.v2x.aux import LDMentry
from opencda.customize.v2x.aux import newLDMentry
from opencda.customize.v2x.aux import Perception
from opencda.customize.v2x.LDMutils import compute_IoU
from opencda.customize.v2x.LDMutils import PO_kalman_filter
from opencda.customize.v2x.LDMutils import get_o3d_bbx
from opencda.customize.platooning.states import FSM
import time


class ExtendedVehicleManager(VehicleManager):
    """
    A class manager to embed different modules with vehicle together.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle. We need this class to spawn our gnss and imu sensor.

    config_yaml : dict
        The configuration dictionary of this CAV.

    application : list
        The application category, currently support:['single','platoon'].

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World. This is used for V2X communication simulation.

    current_time : str
        Timestamp of the simulation beginning, used for data dumping.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    v2x_manager : opencda object
        The current V2X manager.

    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    agent : opencda object
        The current carla agent that handles the basic behavior
         planning of ego vehicle.

    controller : opencda object
        The current control manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    def __init__(
            self,
            vehicle,
            config_yaml,
            application,
            carla_map,
            cav_world,
            current_time='',
            data_dumping=False,
            pldm=False,
            log_dir=None,
            ms_vanet=None):

        super(ExtendedVehicleManager, self).__init__(vehicle,
                                                     config_yaml,
                                                     application,
                                                     carla_map,
                                                     cav_world,
                                                     current_time,
                                                     data_dumping)
        self.PLDM = None
        self.pldm = pldm
        self.platooning = False
        self.log_dir = log_dir
        self.ms_vanet_agent = ms_vanet
        self.LDM_ids = set(range(1, 256))  # ID pool
        self.time = 0.0
        # if self.perception_manager.lidar:
        # self.o3d_vis = o3d_visualizer_init(vehicle.id * 1000)
        self.lidar_visualize = config_yaml['sensing']['perception']['lidar']['visualize']
        self.cav_world = cav_world
        if log_dir:
            if pldm:
                self.file = self.log_dir + '/local_t_PLDM' + str(vehicle.id) + '.csv'
            else:
                self.file = self.log_dir + '/local_t_LDM' + str(vehicle.id) + '.csv'
            with open(self.file, 'w', newline='') as logfile:
                writer = csv.writer(logfile)
                writer.writerow(['Timestamp', 'detection', 'localFusion', 'detectedPOs', 'trackedPOs'])
        else:
            self.file = None
        # self.LDM_CPM_idMap = {}
        self.ldm_mutex = threading.Lock()

        if self.pldm:
            # self.PLDM = PLDM(self, self.v2xAgent, visualize=True, log=False)
            self.pldm_mutex = threading.Lock()
            self.platooning = True
        elif 'platooning' in application:
            self.platooning = True

        self.v2xAgent = None
        if self.ms_vanet_agent is None:
            self.v2xAgent = V2XAgent(self,
                                     ldm_mutex=self.ldm_mutex,
                                     AMQPbroker="127.0.0.1:5672",
                                     PLDM=pldm,
                                     log_dir=self.log_dir)
            self.agent.v2xAgent = weakref.ref(self.v2xAgent)()

        if 'platooning' in application and self.ms_vanet_agent is None:
            self.platooning = True
            if config_yaml['v2x']['platoon_init_pos'] == 1:
                self.agent.v2xAgent.pcService.status = FSM.LEADING_MODE
                if pldm:
                    self.agent.v2xAgent.pldmService.setLeader(True)
            elif config_yaml['v2x']['platoon_init_pos'] > 1:
                self.agent.v2xAgent.pcService.status = FSM.MAINTINING
            self.agent.v2xAgent.pcService.platoon_position = config_yaml['v2x']['platoon_init_pos']

        if not self.pldm:
            self.LDM = LDM(self, self.v2xAgent, visualize=self.lidar_visualize)

    def get_time_ms(self):
        return self.map_manager.world.get_snapshot().elapsed_seconds * 1000

    def get_time(self):
        return self.map_manager.world.get_snapshot().elapsed_seconds

    def update_info_LDM(self):
        # localization
        file_timestamp = time.time_ns() / 1000
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # print("Vehicle ", self.vehicle.id, " speed: ", ego_spd)
        # object detection
        objects = self.perception_manager.detect(ego_pos)
        detected_n = len(objects['vehicles'])
        file_detection = (time.time_ns() / 1000) - file_timestamp
        # ------------------LDM patch------------------------
        self.time = self.map_manager.world.get_snapshot().elapsed_seconds
        if (self.PLDM is not None) and \
                (self.v2xAgent.pldmService.recv_plu > 1 or self.v2xAgent.pldmService.leader):
            # if self.PLDM is not None:
            self.pldm_mutex.acquire()
            self.PLDM.updatePLDM(self.translateDetections(objects))
            objects = self.PLDM.PLDM2OpencdaObj(objects['traffic_lights'])
            self.pldm_mutex.release()
        elif not self.pldm:
            self.ldm_mutex.acquire()
            self.LDM.updateLDM(self.translateDetections(objects))
            objects = self.LDM.LDM2OpencdaObj(objects['traffic_lights'])
            self.ldm_mutex.release()
        file_localFusion = ((time.time_ns() / 1000) - file_detection - file_timestamp)

        if self.file:
            with open(self.file, 'a', newline='') as logfile:
                writer = csv.writer(logfile)
                if self.pldm and self.PLDM is not None:
                    writer.writerow([file_timestamp, file_detection, file_localFusion, detected_n, len(self.PLDM.PLDM)])
                else:
                    writer.writerow([file_timestamp, file_detection, file_localFusion, detected_n, self.LDM.get_LDM_size()])
        # ---------------------------------------------------

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # this is required by safety manager
        safety_input = {'ego_pos': ego_pos,
                        'ego_speed': ego_spd,
                        'objects': objects,
                        'carla_map': self.carla_map,
                        'world': self.vehicle.get_world(),
                        'static_bev': self.map_manager.static_bev}
        self.safety_manager.update_info(safety_input)

        # update ego position and speed to v2x manager,
        # and then v2x manager will search the nearby cavs
        # self.v2x_manager.update_info(ego_pos, ego_spd)
        if self.v2xAgent is not None:
            self.v2xAgent.tick()

        if self.platooning:
            objects = self.clean_platoon_whitelist(objects)

        self.agent.update_information(ego_pos, ego_spd, objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)

    def clean_platoon_whitelist(self, objects):
        ret_objects = {'vehicles': [], 'traffic_lights': []}
        for obj in objects['vehicles']:
            if obj.carla_id in self.v2xAgent.pcService.platoon_list.values():
                continue
            ret_objects['vehicles'].append(obj)
        return ret_objects
    def get_context(self):
        if self.pldm and self.PLDM is not None:
            return self.PLDM.getPLDM_perceptions()
        else:
            return self.LDM.getLDM_perceptions()

    def translateDetections(self, object_list):
        ego_pos, ego_spd, objects = self.getInfo()
        returnedObjects = []
        for obj in object_list['vehicles']:
            if obj.carla_id == -1:
                continue  # If object can't be matched with a CARLA vehicle, we ignore it
            dist = math.sqrt(
                math.pow((obj.location.x - ego_pos.location.x), 2) + math.pow((obj.location.y - ego_pos.location.y),
                                                                              2))
            LDMobj = Perception(obj.location.x,
                                obj.location.y,
                                obj.bounding_box.extent.x * 2,
                                obj.bounding_box.extent.y * 2,
                                self.time,
                                obj.confidence)
            LDMobj.xSpeed = obj.velocity.x
            LDMobj.ySpeed = obj.velocity.y
            returnedObjects.append(LDMobj)
        return {'vehicles': returnedObjects}

    def getInfo(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # object detection
        objects = self.perception_manager.objects

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        return ego_pos, ego_spd, objects

    def LDM_to_lidarObjects(self):
        lidarObjects = []
        for ID, LDMobj in self.LDM.items():
            lidarObjects.append(LDMobj)  # return last sample of each object in LDM
        return {'vehicles': lidarObjects}

    def matchLDMobject(self, object):
        matched = False
        matchedId = -1
        for ID, LDMobj in self.LDM.items():
            x, y, w, l, vx, vy = LDMobj.kalman_filter.predict()

            LDMcurrX = LDMobj.perception.xPosition
            LDMcurrY = LDMobj.perception.yPosition
            LDMcurrbbx = LDMobj.perception.o3d_bbx
            if self.time > LDMobj.perception.timestamp:
                LDMcurrX += (self.time - LDMobj.perception.timestamp) * LDMobj.perception.xSpeed
                LDMcurrY += (self.time - LDMobj.perception.timestamp) * LDMobj.perception.ySpeed
                LDMcurrbbx = self.LDMobj_to_o3d_bbx(LDMobj.perception)
            dist = math.sqrt(math.pow((object.xPosition - LDMcurrX), 2) + math.pow((object.yPosition - LDMcurrY), 2))
            iou = compute_IoU(object.o3d_bbx, LDMcurrbbx)
            if dist < 3 or iou > 0.5:
                matchedId = ID
                break
        return matchedId

    def LDMobj_to_o3d_bbx(self, LDMobj):
        # o3d bbx test
        lidarPos = self.perception_manager.lidar.sensor.get_transform()
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

    def obj_to_o3d_bbx(self, obj):
        lidarPos = self.perception_manager.lidar.sensor.get_transform()
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

    def LDMobj_to_min_max(self, LDMobj):
        # o3d bbx test
        lidarPos = self.perception_manager.lidar.sensor.get_transform()
        objRelPos = np.array([-1 * (LDMobj.xPosition - lidarPos.location.x),
                              LDMobj.yPosition - lidarPos.location.y,
                              1.5 - lidarPos.location.z])
        min_arr = objRelPos - np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
        max_arr = objRelPos + np.array([LDMobj.width / 2, LDMobj.length / 2, 0.75])
        return min_arr, max_arr

    def appendObject(self, obj, id):
        if self.vehicle.id not in self.LDM[id].perceivedBy:
            self.LDM[id].perceivedBy.append(self.vehicle.id)
        if obj.timestamp < self.LDM[id].getLatestPoint().timestamp:
            return
        # Update kalman filter
        # x, y, w, l, vx, vy = self.LDM[id].kalman_filter.update(obj.xPosition, obj.yPosition, obj.width, obj.length)
        # obj.xPosition = x
        # obj.yPosition = y
        # obj.width = w
        # obj.length = l
        # obj.xSpeed = vx
        # obj.ySpeed = vy
        # For debugging predicted speed computation -----------------------------------
        # with open(self.kfile, 'a', newline='') as logfile:
        #     writer = csv.writer(logfile)
        #     writer.writerow([obj.timestamp, id, obj.xPosition, obj.yPosition, x, y, obj.xSpeed,
        #                      obj.ySpeed, vx, vy, math.degrees(math.atan2(obj.ySpeed, obj.xSpeed)),
        #                      math.degrees(math.atan2(vy, vx))])
        # -----------------------------------------------------------------------------

        # timeDiff = obj.timestamp - self.LDM[id].pathHistory[0].timestamp
        # if timeDiff >= 0.05:
        #     xSpeed = (obj.xPosition - self.LDM[id].pathHistory[0].xPosition) / timeDiff
        #     ySpeed = (obj.yPosition - self.LDM[id].pathHistory[0].yPosition) / timeDiff
        # else:
        #     xSpeed = self.LDM[id].perception.xSpeed
        #     ySpeed = self.LDM[id].perception.ySpeed
        #
        # obj.xSpeed = xSpeed
        # obj.ySpeed = ySpeed

        # Compute the estimated heading angle
        obj.heading = math.degrees(math.atan2(obj.ySpeed, obj.xSpeed))

        if self.LDM[id].detected is True:
            # If this entry is of a connected vehicle
            obj.width = self.LDM[id].perception.width
            obj.length = self.LDM[id].perception.length
        else:
            width_max = obj.width
            length_max = obj.length
            for prev_obj in self.LDM[id].pathHistory:
                if prev_obj.width > width_max:
                    width_max = prev_obj.width
                if length_max < prev_obj.length <= 2.1:
                    length_max = prev_obj.length
            obj.width = width_max
            obj.length = length_max
            # Compute the average width and length of all objects in the deque
            # width_sum = obj.width
            # length_sum = obj.length
            # for prev_obj in self.LDM[obj.id]:
            #     width_sum += prev_obj.width
            #     length_sum += prev_obj.length
            # avg_width = width_sum / (len(self.LDM[obj.id]) + 1)
            # avg_length = length_sum / (len(self.LDM[obj.id]) + 1)
            # obj.width = avg_width
            # obj.length = avg_length

        obj.o3d_bbx = self.LDMobj_to_o3d_bbx(obj)
        self.LDM[id].insertPerception(obj)
