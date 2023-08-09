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
from opencda.customize.v2x.aux import LDMObject
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
import opencda.core.sensing.perception.sensor_transformation as st
import csv
import weakref

from opencda.customize.v2x.v2x_agent import V2XAgent
from opencda.customize.v2x.PLDM import PLDM
from opencda.customize.v2x.aux import LDMentry
from opencda.customize.v2x.aux import newLDMentry
from opencda.customize.v2x.aux import Perception
from opencda.customize.v2x.LDMutils import LDM2OpencdaObj
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
            log_dir=None):

        super(ExtendedVehicleManager, self).__init__(vehicle,
                                                     config_yaml,
                                                     application,
                                                     carla_map,
                                                     cav_world,
                                                     current_time,
                                                     data_dumping)
        self.LDM = {}
        self.PLDM = None
        self.pldm = pldm
        self.log_dir = log_dir
        self.LDM_ids = set(range(1, 256))  # ID pool
        self.time = 0.0
        if self.perception_manager.lidar:
            self.o3d_vis = o3d_visualizer_init(vehicle.id * 1000)
        self.cav_world = cav_world
        if pldm:
            self.file = self.log_dir + '/local_t_PLDM' + str(vehicle.id) + '.csv'
        else:
            self.file = self.log_dir + '/local_t_LDM' + str(vehicle.id) + '.csv'
        with open(self.file, 'w', newline='') as logfile:
            writer = csv.writer(logfile)
            writer.writerow(['Timestamp', 'detection', 'localFusion', 'detectedPOs', 'trackedPOs'])
        # self.LDM_CPM_idMap = {}
        self.ldm_mutex = threading.Lock()

        if self.pldm:
            # self.PLDM = PLDM(self, self.v2xAgent, visualize=True, log=False)
            self.pldm_mutex = threading.Lock()
            self.platooning = False

        self.v2xAgent = V2XAgent(self,
                                 ldm_mutex=self.ldm_mutex,
                                 AMQPbroker="127.0.0.1:5672",
                                 PLDM=pldm,
                                 log_dir=self.log_dir)
        self.agent.v2xAgent = weakref.ref(self.v2xAgent)()
        if config_yaml['v2x']['platoon_init_pos'] == 1:
            self.agent.v2xAgent.pcService.status = FSM.LEADING_MODE
            if pldm:
                self.agent.v2xAgent.pldmService.setLeader(True)
        elif config_yaml['v2x']['platoon_init_pos'] > 1:
            self.agent.v2xAgent.pcService.status = FSM.MAINTINING
        self.agent.v2xAgent.pcService.platoon_position = config_yaml['v2x']['platoon_init_pos']

    def get_time_ms(self):
        return self.time * 1000

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
                (self.v2xAgent.pldmService.recv_plu != 0 or self.v2xAgent.pldmService.leader):
            self.pldm_mutex.acquire()
            self.PLDM.updatePLDM(self.translateDetections(objects))
            objects = LDM2OpencdaObj(self, self.PLDM.PLDM, objects['traffic_lights'])
            self.pldm_mutex.release()
        else:
            self.ldm_mutex.acquire()
            self.update_LDM(self.translateDetections(objects))
            objects = self.LDM2OpencdaObj(objects['traffic_lights'])
            self.ldm_mutex.release()
        file_localFusion = ((time.time_ns() / 1000) - file_detection - file_timestamp)
        with open(self.file, 'a', newline='') as logfile:
            writer = csv.writer(logfile)
            if self.pldm and self.PLDM is not None:
                writer.writerow([file_timestamp, file_detection, file_localFusion, detected_n, len(self.PLDM.PLDM)])
            else:
                writer.writerow([file_timestamp, file_detection, file_localFusion, detected_n, len(self.LDM)])
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

        self.v2xAgent.tick()

        self.agent.update_information(ego_pos, ego_spd, objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)

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
                                (100 - dist) if dist < 100 else 0)
            LDMobj.xSpeed = obj.velocity.x
            LDMobj.ySpeed = obj.velocity.y
            returnedObjects.append(LDMobj)
        return {'vehicles': returnedObjects}

    def cleanDuplicates(self):
        duplicates = []
        for ID, LDMobj in self.LDM.items():
            matchedId = self.matchLDMobject(LDMobj.perception)
            if matchedId != -1 and matchedId != LDMobj.id:
                if LDMobj.detected is True and self.LDM[matchedId].detected is False:
                    duplicates.append(matchedId)
                elif LDMobj.detected is True and LDMobj.perception.timestamp > self.LDM[matchedId].perception.timestamp:
                    duplicates.append(matchedId)
                elif LDMobj.detected is True and \
                        (LDMobj.perception.width + LDMobj.perception.length) > (
                        self.LDM[matchedId].perception.width + self.LDM[matchedId].perception.length):
                    duplicates.append(matchedId)
                elif LDMobj.detected is False:
                    duplicates.append(matchedId)
                else:
                    duplicates.append(ID)
        deleted = []
        for ID in duplicates:
            if ID not in deleted:
                del self.LDM[ID]
                deleted.append(ID)  # In case we have duplicates in the 'duplicates' list
                self.LDM_ids.add(ID)

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

    def LDM2OpencdaObj(self, trafficLights):
        # LDM = self.getLDM()
        retObjects = []
        for ID, LDMObject in self.LDM.items():
            corner = np.asarray(LDMObject.perception.o3d_bbx.get_box_points())
            # covert back to unreal coordinate
            corner[:, :1] = -corner[:, :1]
            corner = corner.transpose()
            # extend (3, 8) to (4, 8) for homogenous transformation
            corner = np.r_[corner, [np.ones(corner.shape[1])]]
            # project to world reference
            corner = st.sensor_to_world(corner, self.perception_manager.lidar.sensor.get_transform())
            corner = corner.transpose()[:, :3]
            object = ObstacleVehicle(corner, LDMObject.perception.o3d_bbx)
            object.carla_id = LDMObject.id
            retObjects.append(object)

        return {'vehicles': retObjects, 'traffic_lights': trafficLights}

    def getLDM(self):
        retLDM = {}
        for ID, LDMobj in self.LDM.items():
            retLDM[ID] = LDMobj[(len(self.LDM[ID]) - 1)]  # return last sample of each object in LDM

        return retLDM

    def getCPM(self):
        retLDM = {}
        for ID, LDMobj in self.LDM.items():
            if len(self.LDM[ID].pathHistory) < 9:
                continue
            retLDM[ID] = LDMobj

        return retLDM

    def LDM_to_lidarObjects(self):
        lidarObjects = []
        for ID, LDMobj in self.LDM.items():
            lidarObjects.append(LDMobj)  # return last sample of each object in LDM
        return {'vehicles': lidarObjects}

    def CAMfusion(self, CAMobject):
        newLDMobject = []
        matched = False
        CAMobject.connected = True
        match = self.matchLDMobject(CAMobject)
        if match != -1:
            CAMobject.id = match
        newLDMobject.append(CAMobject)
        newObjects = {'vehicles': newLDMobject}
        self.update_LDM(newObjects, True)
        return CAMobject.id

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

    def CPMfusion(self, object_List, fromID):
        # Try to match CPM objects with LDM ones
        # If we match an object, we perform fusion averaging the bbx
        # If can't match the object we append it to the LDM as a new object
        newLDMobjects = []
        for CPMobj in object_List:
            CPMobj.detected = False
            CPMobj.detectedBy = fromID
            # print('[CPM fusion] '+str(CPMobj.id) + ' speed: ' + str(CPMobj.xSpeed) + ',' + str(CPMobj.ySpeed))
            if self.time > CPMobj.timestamp:
                # If it's an old perception, we need to predict its current position
                CPMobj.xPosition += CPMobj.xSpeed * (self.time - CPMobj.timestamp)
                CPMobj.yPosition += CPMobj.ySpeed * (self.time - CPMobj.timestamp)
            matchedID = self.matchLDMobject(CPMobj)
            if matchedID != -1:
                LDMobj = self.getLDM()[matchedID]
                newLDMobj = LDMObject(CPMobj.id,
                                      CPMobj.xPosition,
                                      CPMobj.yPosition,
                                      CPMobj.length,
                                      CPMobj.width,
                                      CPMobj.timestamp,
                                      CPMobj.confidence)
                newLDMobj.xSpeed = CPMobj.xSpeed
                newLDMobj.ySpeed = CPMobj.ySpeed
                newLDMobj.heading = CPMobj.heading
                newLDMobj.id = matchedID
                if self.LDM[matchedID][(len(self.LDM[matchedID]) - 1)].detectedBy != fromID:
                    # Compute weights depending on the timestamps and confidence (~distance from detecting vehicle)
                    weightLDM = (LDMobj.timestamp / (CPMobj.timestamp + LDMobj.timestamp)) * \
                                (LDMobj.confidence / (LDMobj.confidence + CPMobj.confidence))
                    weightCPM = (CPMobj.timestamp / (CPMobj.timestamp + LDMobj.timestamp)) * \
                                (CPMobj.confidence / (LDMobj.confidence + CPMobj.confidence))
                    # Wish there was a cleaner way to do this
                    # newLDMobj.xPosition = (currX * weightCPM + CPMobj.xPosition * weightLDM) / (weightLDM + weightCPM)
                    # newLDMobj.yPosition = (currY * weightCPM + CPMobj.yPosition * weightLDM) / (weightLDM + weightCPM)
                    # newLDMobj.xSpeed = (CPMobj.xSpeed * weightCPM + LDMobj.xSpeed * weightLDM) / (weightLDM + weightCPM)
                    # newLDMobj.ySpeed = (CPMobj.ySpeed * weightCPM + LDMobj.ySpeed * weightLDM) / (weightLDM + weightCPM)
                    newLDMobj.width = (CPMobj.width * weightCPM + LDMobj.width * weightLDM) / (weightLDM + weightCPM)
                    newLDMobj.length = (CPMobj.length * weightCPM + LDMobj.length * weightLDM) / (weightLDM + weightCPM)
                    # newLDMobj.heading = (CPMobj.heading * weightCPM + LDMobj.heading * weightLDM) / (
                    #         weightLDM + weightCPM)
                    # newLDMobj.confidence = (CPMobj.confidence + LDMobj.confidence) / 2
                    # newLDMobj.timestamp = self.time
                # print('[CPM after fusion] '+str(newLDMobj.id) + ' speed: ' + str(newLDMobj.xSpeed) + ',' + str(newLDMobj.ySpeed))
                newLDMobjects.append(newLDMobj)
            else:
                # We add the CPM object as a new perception
                newLDMobjects.append(CPMobj)
        newObjects = {'vehicles': newLDMobjects}
        self.update_LDM(newObjects, True)

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

    def match_LDM_local(self, object_list):
        if len(self.LDM) != 0:
            IoU_map = np.zeros((len(self.LDM), len(object_list)), dtype=np.float32)
            i = 0
            ldm_ids = []
            for ID, LDMobj in self.LDM.items():
                for j in range(len(object_list)):
                    obj = object_list[j]
                    object_list[j].o3d_bbx = self.LDMobj_to_o3d_bbx(obj)
                    # LDMpredX, LDMpredY, LDMpredXe, LDMpredYe, \
                    #     LDMpredXspeed, LDMpredYspeed = LDMobj.kalman_filter.predict()
                    LDMpredX = LDMobj.perception.xPosition
                    LDMpredY = LDMobj.perception.yPosition
                    LDMpredbbx = LDMobj.perception.o3d_bbx
                    if self.time > LDMobj.perception.timestamp:
                        LDMpredX += (self.time - LDMobj.perception.timestamp) * LDMobj.perception.xSpeed
                        LDMpredY += (self.time - LDMobj.perception.timestamp) * LDMobj.perception.ySpeed
                        LDMpredbbx = self.LDMobj_to_o3d_bbx(LDMobj.perception)
                    LDMpredbbx = get_o3d_bbx(self, LDMpredX, LDMpredY, LDMobj.perception.width,
                                             LDMobj.perception.length)

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

    def update_LDM(self, object_list, CPM=False):
        updated_ids = []
        IoU_map, new, matched, ldm_ids = self.match_LDM_local(object_list['vehicles'])
        for j in range(len(object_list['vehicles'])):
            obj = object_list['vehicles'][j]
            obj.o3d_bbx = self.LDMobj_to_o3d_bbx(obj)
            # matchedID = self.matchLDMobject(obj)
            if IoU_map is not None:
                matchedObj = matched[np.where(new == j)[0]]
                if IoU_map[matchedObj, j] >= 0:
                    updated_ids.append(ldm_ids[matchedObj[0]])
                    self.appendObject(obj, ldm_ids[matchedObj[0]])
                    continue
            newID = self.LDM_ids.pop()
            self.LDM[newID] = newLDMentry(obj, newID, detected=True, onSight=True)
            self.LDM[newID].kalman_filter = PO_kalman_filter()
            self.LDM[newID].kalman_filter.init_step(obj.xPosition, obj.yPosition, obj.width, obj.length)

        # # Update the position of not updated objects
        for ID, LDMobj in self.LDM.items():
            if LDMobj.perception.timestamp < self.time:
                LDMobj.perception.xPosition += \
                    (self.time - LDMobj.perception.timestamp) * LDMobj.perception.xSpeed
                LDMobj.perception.yPosition += \
                    (self.time - LDMobj.perception.timestamp) * LDMobj.perception.ySpeed
                LDMobj.perception.o3d_bbx = self.LDMobj_to_o3d_bbx(LDMobj.perception)
                LDMobj.perception.timestamp = self.time

        # Delete old perceptions
        if self.time > 2.0:
            T = self.time - 2.0
            old_ids = [ID for ID, LDMobj in self.LDM.items() if LDMobj.getLatestPoint().timestamp <= T]
            for ID in old_ids:
                del self.LDM[ID]
                self.LDM_ids.add(ID)

        # Clean possible duplicates
        if len(self.LDM) != 0:
            self.clean_duplicates()

        # LDM visualization in lidar view
        showObjects = self.LDM_to_lidarObjects()
        gt = self.perception_manager.getGTobjects()
        if self.perception_manager.lidar:
            while self.perception_manager.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.perception_manager.lidar.data, self.perception_manager.lidar.o3d_pointcloud)
            o3d_visualizer_showLDM(
                self.o3d_vis,
                self.perception_manager.count,
                self.perception_manager.lidar.o3d_pointcloud,
                showObjects,
                gt)

    def clean_duplicates(self):
        objects = [obj.perception for obj in self.LDM.values()]
        IoU_map, new, matched, ldm_ids = self.match_LDM_local(objects)
        indices_to_delete = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if IoU_map[i][j] >= 0 and self.LDM[ldm_ids[j]].detected:
                    indices_to_delete.append(j)
        indices_to_delete = list(set(indices_to_delete))

        for i in indices_to_delete:
            del self.LDM[ldm_ids[i]]
            self.LDM_ids.add(ldm_ids[i])

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
