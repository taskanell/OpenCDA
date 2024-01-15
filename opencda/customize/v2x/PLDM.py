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

from opencda.customize.v2x.LDMutils import LDM_to_lidarObjects
from opencda.customize.v2x.LDMutils import matchLDMobject
from opencda.customize.v2x.LDMutils import LDMobj_to_o3d_bbx
from opencda.customize.v2x.LDMutils import get_o3d_bbx
from opencda.customize.v2x.LDMutils import compute_IoU
from scipy.optimize import linear_sum_assignment as linear_assignment
from opencda.customize.v2x.aux import newPLDMentry
from opencda.customize.v2x.LDMutils import PO_kalman_filter
from opencda.customize.v2x.LDMutils import obj_to_o3d_bbx
from opencda.customize.v2x.aux import ColorGradient


class PLDM(object):
    def __init__(
            self,
            cav,
            V2Xagent,
            leader=False,
            visualize=True,
            log=True
    ):
        self.PLDM = {}
        self.CPM_buffer = {}
        self.PLDM_ids = set(range(1, 256))  # ID pool
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.pmMap = {}
        self.leader = leader
        self.colorGradient = ColorGradient(10)
        if visualize:
            self.o3d_vis = o3d_visualizer_init(cav.vehicle.id * 100)
        # if log:
        #     self.file = '/home/carlos/speed_logs' + str(cav.vehicle.id) + 'PLDM.csv'
        #     with open(self.file, 'w', newline='') as logfile:
        #         writer = csv.writer(logfile)
        #         writer.writerow(
        #             ['Timestamp', 'vID', 'cX', 'cY', 'PxSpeed', 'PySpeed', 'GTxSpeed', 'GTySpeed', 'Heading'])

    def match_PLDM(self, object_list):
        if len(self.PLDM) != 0:
            IoU_map = np.zeros((len(self.PLDM), len(object_list)), dtype=np.float32)
            i = 0
            ldm_ids = []
            for ID, PLDMobj in self.PLDM.items():
                for j in range(len(object_list)):
                    obj = object_list[j]
                    object_list[j].o3d_bbx = self.cav.LDMobj_to_o3d_bbx(obj)
                    LDMpredX = PLDMobj.perception.xPosition
                    LDMpredY = PLDMobj.perception.yPosition
                    # if self.cav.time > PLDMobj.perception.timestamp:
                    #     LDMpredX += (self.cav.time - PLDMobj.perception.timestamp) * PLDMobj.perception.xSpeed
                    #     LDMpredY += (self.cav.time - PLDMobj.perception.timestamp) * PLDMobj.perception.ySpeed
                    LDMpredbbx, LDMpredline_set = get_o3d_bbx(self.cav, LDMpredX, LDMpredY, PLDMobj.perception.width,
                                                              PLDMobj.perception.length, PLDMobj.perception.yaw)

                    dist = math.sqrt(
                        math.pow((obj.xPosition - LDMpredX), 2) + math.pow(
                            (obj.yPosition - LDMpredY), 2))
                    iou = compute_IoU(LDMpredbbx, object_list[j].o3d_bbx)
                    if iou > 0:
                        IoU_map[i, j] = iou
                    elif dist < 4:  # if dist < 3 --> IoU_map[i, j] = 0
                        IoU_map[i, j] = dist - 1000
                    else:
                        IoU_map[i, j] = -1000
                i += 1
                ldm_ids.append(ID)
            matched, new = linear_assignment(-IoU_map)
            return IoU_map, new, matched, ldm_ids
        return None, None, None, None

    def updatePLDM(self, object_list):
        # Predict position of current PLDM tracks before attempting to match
        for ID, PLDMobj in self.PLDM.items():
            # diff = self.cav.time - LDMobj.perception.timestamp
            self.PLDM[ID].onSight = False  # It will go back to True when appending if we match it
            PLDMobj.perception.xPosition, \
                PLDMobj.perception.yPosition, \
                PLDMobj.perception.xSpeed, \
                PLDMobj.perception.ySpeed, \
                PLDMobj.perception.xacc, \
                PLDMobj.perception.yacc = self.PLDM[ID].kalman_filter.predict(self.cav.get_time_ms())
            PLDMobj.perception.o3d_bbx, PLDMobj.perception.line_set = self.cav.LDMobj_to_o3d_bbx(PLDMobj.perception)
            PLDMobj.perception.timestamp = self.cav.time

        IoU_map, new, matched, pldm_ids = self.match_PLDM(object_list['vehicles'])
        for j in range(len(object_list['vehicles'])):
            obj = object_list['vehicles'][j]
            if IoU_map is not None:
                matchedObj = matched[np.where(new == j)[0]]
                if IoU_map[matchedObj, j] >= 0:
                    if self.PLDM[pldm_ids[matchedObj[0]]].detected:
                        # If we match a PO that this PM is assigned with or is still a newPO, update PLDM
                        if (self.PLDM[pldm_ids[matchedObj[0]]].assignedPM == self.cav.vehicle.id) or \
                                self.PLDM[pldm_ids[matchedObj[0]]].newPO:
                            self.appendObject(obj, pldm_ids[matchedObj[0]])
                        else:
                            # If we match a PO that is assigned to other PM, we add perception to CPM buffer
                            self.CPM_buffer[pldm_ids[matchedObj[0]]] = newPLDMentry(obj,
                                                                                    matchedObj[0],
                                                                                    detected=True,
                                                                                    onSight=True)
                            # Update the perceivedBy list
                            if self.cav.vehicle.id not in self.PLDM[pldm_ids[matchedObj[0]]].perceivedBy:
                                self.PLDM[pldm_ids[matchedObj[0]]].perceivedBy.append(self.cav.vehicle.id)
                                self.PLDM[pldm_ids[matchedObj[0]]].onSight = True
                    continue
            # we are detecting a new object
            newID = self.PLDM_ids.pop()
            obj.o3d_bbx = self.cav.LDMobj_to_o3d_bbx(obj)
            self.PLDM[newID] = newPLDMentry(obj, newID, detected=True, onSight=True)
            if self.leader:
                self.PLDM[newID].newPO = False
                self.PLDM[newID].assignedPM = self.cav.vehicle.id
            else:
                self.PLDM[newID].newPO = True
                self.PLDM[newID].assignedPM = None
            self.PLDM[newID].kalman_filter = PO_kalman_filter()
            self.PLDM[newID].kalman_filter.init_step(obj.xPosition,
                                                     obj.yPosition,
                                                     vx=self.cav.vehicle.get_velocity().x,
                                                     vy=self.cav.vehicle.get_velocity().y,
                                                     ax=self.cav.vehicle.get_acceleration().x,
                                                     ay=self.cav.vehicle.get_acceleration().y)

        # Delete old perceptions
        if self.cav.time > 2.0:
            T = self.cav.time - 2.0
            old_ids = [ID for ID, PLDMobj in self.PLDM.items() if PLDMobj.getLatestPoint().timestamp <= T]
            for ID in old_ids:
                del self.PLDM[ID]
                self.PLDM_ids.add(ID)
            if self.cav.time > 2.0:
                T = self.cav.time - 0.5
                old_ids = [ID for ID, LDMobj in self.PLDM.items() if LDMobj.getLatestPoint().timestamp <= T
                           and (self.cav.vehicle.id not in LDMobj.perceivedBy or not LDMobj.tracked)]
                for ID in old_ids:
                    del self.PLDM[ID]
                    self.PLDM_ids.add(ID)
            # Also from CPM buffer
            T = self.cav.time - 0.5
            old_ids = [ID for ID, PLDMobj in self.CPM_buffer.items() if PLDMobj.getLatestPoint().timestamp <= T]
            for ID in old_ids:
                del self.CPM_buffer[ID]

        self.clean_duplicates()

        # LDM visualization in lidar view
        showObjects = LDM_to_lidarObjects(self)
        gt = self.cav.perception_manager.getGTobjects()
        if self.cav.perception_manager.lidar:
            while self.cav.perception_manager.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.cav.perception_manager.lidar.data,
                                  self.cav.perception_manager.lidar.o3d_pointcloud)
            if self.cav.lidar_visualize:
                o3d_visualizer_showLDM(
                    self.o3d_vis,
                    self.cav.perception_manager.count,
                    self.cav.perception_manager.lidar.o3d_pointcloud,
                    showObjects,
                    gt)

    def clean_duplicates(self):
        objects = [obj.perception for obj in self.PLDM.values()]
        IoU_map, new, matched, pldm_ids = self.match_PLDM(objects)
        indices_to_delete = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if IoU_map[i][j] > 0 and self.PLDM[pldm_ids[j]].detected:
                    indices_to_delete.append(j)
        indices_to_delete = list(set(indices_to_delete))

        for i in indices_to_delete:
            del self.PLDM[pldm_ids[i]]
            self.PLDM_ids.add(pldm_ids[i])

    def appendObject(self, obj, id):
        if self.cav.vehicle.id not in self.PLDM[id].perceivedBy:
            self.PLDM[id].perceivedBy.append(self.cav.vehicle.id)
        if obj.timestamp <= self.PLDM[id].getLatestPoint().timestamp:
            return

        self.PLDM[id].onSight = True
        # Compute the estimated heading angle
        heading = math.degrees(math.atan2(obj.ySpeed, obj.xSpeed))
        obj.heading = heading

        if self.PLDM[id].detected is False:
            # If this entry is of a connected vehicle
            # obj.connected = True  # We do this to take the width and length from CAM always
            obj.width = self.PLDM[obj.id].perception.width
            obj.length = self.PLDM[obj.id].perception.length
        else:
            width_max = obj.width
            length_max = obj.length
            for prev_obj in self.PLDM[id].pathHistory:
                if prev_obj.width > width_max:
                    width_max = prev_obj.width
                if length_max < prev_obj.length <= 2.1:
                    length_max = prev_obj.length
            obj.width = width_max
            obj.length = length_max

        obj.o3d_bbx, obj.line_set = LDMobj_to_o3d_bbx(self.cav, obj)

        x, y, vx, vy, ax, ay = self.PLDM[id].kalman_filter.update(obj.xPosition, obj.yPosition, self.cav.get_time_ms())
        # print('KFupdate: ', "x: ", x, ",y: ", y, ",vx: ", vx, ",vy: ", vy, ",ax: ", ax, ",ay: ", ay)
        obj.xPosition = x
        obj.yPosition = y
        obj.xSpeed = vx
        obj.ySpeed = vy
        obj.xacc = ax
        obj.yacc = ay

        # print('[LDM append] '+str(obj.id) + ' speed: ' + str(obj.xSpeed) + ',' + str(obj.ySpeed))

        self.PLDM[id].insertPerception(obj)

    def cleanDuplicates(self):
        # simpleLDM = self.getLDM()
        duplicates = []
        for ID, LDMobj in self.PLDM.items():
            matchedId = matchLDMobject(self, LDMobj)
            if matchedId != -1:
                if LDMobj.PLU is True and self.PLDM[matchedId].PLU is False:
                    duplicates.append(matchedId)
                elif LDMobj.detected is True and self.PLDM[matchedId].detected is False:
                    duplicates.append(matchedId)
                elif LDMobj.detected is True and LDMobj.timestamp > self.PLDM[matchedId].timestamp:
                    duplicates.append(matchedId)
                elif LDMobj.detected is True and \
                        (LDMobj.width + LDMobj.length) > (self.PLDM[matchedId].width + self.PLDM[matchedId].length):
                    duplicates.append(matchedId)
                elif LDMobj.connected is True:
                    duplicates.append(matchedId)
                else:
                    duplicates.append(ID)
        deleted = []
        for ID in duplicates:
            if ID not in deleted:
                del self.PLDM[ID]
                deleted.append(ID)  # In case we have duplicates in the 'duplicates' list

    def getCPM(self):
        # return self.CPM_buffer # TODO see why CPM buffer is empty
        t_map = []
        cpm = {}
        for ID, entry in self.PLDM.items():
            if entry.detected and entry.onSight:
                t_map.append((entry.getLatestPoint().timestamp, ID))
        sorted_list = sorted(t_map, key=lambda x: x[0], reverse=True)

        for obj in sorted_list[:10]:
            cpm[obj[1]] = self.PLDM[obj[1]]
        return cpm

    def getPLDM_perceptions(self):
        perceptions = {}
        for ID, LDMobj in self.PLDM.items():
            perceptions[ID] = LDMobj.perception
        return perceptions

    def PLDM2OpencdaObj(self, trafficLights):
        retObjects = []
        for ID, LDMObject in self.PLDM.items():
            corner = np.asarray(LDMObject.perception.o3d_bbx.get_box_points())
            # covert back to unreal coordinate
            corner[:, :1] = -corner[:, :1]
            corner = corner.transpose()
            # extend (3, 8) to (4, 8) for homogenous transformation
            corner = np.r_[corner, [np.ones(corner.shape[1])]]
            # project to world reference
            corner = st.sensor_to_world(corner, self.cav.perception_manager.lidar.sensor.get_transform())
            corner = corner.transpose()[:, :3]
            object = ObstacleVehicle(corner, LDMObject.perception.o3d_bbx)
            object.carla_id = LDMObject.id
            retObjects.append(object)

        return {'vehicles': retObjects, 'traffic_lights': trafficLights}

    def getAllPOs(self):
        POs = []
        for ID, PLDMobj in self.PLDM.items():
            if PLDMobj.detected:
                POs.append(PLDMobj)
        return POs
