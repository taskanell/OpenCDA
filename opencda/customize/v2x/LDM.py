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
from opencda.customize.v2x.LDMutils import compute_IoU_lineSet
from scipy.optimize import linear_sum_assignment as linear_assignment
from opencda.customize.v2x.aux import newLDMentry
from opencda.customize.v2x.aux import Perception
from opencda.customize.v2x.LDMutils import PO_kalman_filter
from opencda.customize.v2x.LDMutils import obj_to_o3d_bbx
from opencda.customize.v2x.aux import ColorGradient


class LDM(object):
    def __init__(
            self,
            cav,
            V2Xagent,
            leader=False,
            visualize=True,
            log=True
    ):
        self.LDM = {}
        self.CPM_buffer = {}
        self.LDM_ids = set(range(1, 65536))  # ID pool
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.pmMap = {}
        self.leader = leader
        self.last_update = 0
        self.recvCPMmap = {}
        self.colorGradient = ColorGradient(10)
        if visualize:
            self.o3d_vis = o3d_visualizer_init(cav.vehicle.id) #* 10)
        # if log:
        #     self.file = '/home/carlos/speed_logs' + str(cav.vehicle.id) + 'LDM.csv'
        #     with open(self.file, 'w', newline='') as logfile:
        #         writer = csv.writer(logfile)
        #         writer.writerow(
        #             ['Timestamp', 'vID', 'cX', 'cY', 'PxSpeed', 'PySpeed', 'GTxSpeed', 'GTySpeed', 'Heading'])

    def match_LDM(self, object_list):
        if len(self.LDM) != 0:
            IoU_map = np.zeros((len(self.LDM), len(object_list)), dtype=np.float32)
            i = 0
            ldm_ids = []
            for ID, LDMobj in self.LDM.items():
                for j in range(len(object_list)):
                    obj = object_list[j]
                    object_list[j].o3d_bbx, object_list[j].line_set = self.cav.LDMobj_to_o3d_bbx(obj)
                    LDMpredX = LDMobj.perception.xPosition
                    LDMpredY = LDMobj.perception.yPosition
                    LDMpredbbx, LDMpredline_set = get_o3d_bbx(self.cav, LDMpredX, LDMpredY, LDMobj.perception.width,
                                                              LDMobj.perception.length, LDMobj.perception.yaw)

                    dist = math.sqrt(
                        math.pow((obj.xPosition - LDMpredX), 2) + math.pow(
                            (obj.yPosition - LDMpredY), 2))
                    iou = compute_IoU(LDMpredbbx, object_list[j].o3d_bbx)
                    try:
                        iou = compute_IoU_lineSet(LDMpredline_set, object_list[j].line_set)
                    except RuntimeError as e:
                        print("Unable to compute the oriented bounding box:", e)
                        pass
                    #if compared objects have diff label have iou -1000
                    if iou > 0 and LDMobj.perception.label == obj.label:
                        IoU_map[i, j] = iou
                    elif dist < 3 and LDMobj.perception.label == obj.label:  # if dist < 3 --> IoU_map[i, j] = 0
                        IoU_map[i, j] = dist - 1000
                    else:
                        IoU_map[i, j] = -1000
                    print(f'IoU value {IoU_map[i, j]} of {obj.id} with {ID} vehicle')
                i += 1
                ldm_ids.append(ID)
            matched, new = linear_assignment(-IoU_map)
            return IoU_map, new, matched, ldm_ids
        return None, None, None, None

    def updateLDM(self, object_list):
        # Predict position of current LDM tracks before attempting to match
        for ID, LDMobj in self.LDM.items():
            # diff = self.cav.time - LDMobj.perception.timestamp
            self.LDM[ID].onSight = False  # It will go back to True when appending if we match it
            LDMobj.perception.xPosition, \
                LDMobj.perception.yPosition, \
                LDMobj.perception.xSpeed, \
                LDMobj.perception.ySpeed, \
                LDMobj.perception.xacc, \
                LDMobj.perception.yacc = self.LDM[ID].kalman_filter.predict(self.cav.get_time_ms())
            LDMobj.perception.o3d_bbx, LDMobj.perception.line_set = self.cav.LDMobj_to_o3d_bbx(LDMobj.perception)
            LDMobj.perception.timestamp = self.cav.time

        IoU_map, new, matched, ldm_ids = self.match_LDM(object_list['vehicles'])
        for j in range(len(object_list['vehicles'])):
            obj = object_list['vehicles'][j]
            obj.o3d_bbx, obj.line_set = self.cav.LDMobj_to_o3d_bbx(obj)
            if IoU_map is not None:
                print (f'IoU map is not None for {obj.id}')
                matchedObj = matched[np.where(new == j)[0]]
                if IoU_map[matchedObj, j] != -1000:
                    print(f"MATCHED OBJECT ID {obj.id}")
                    self.appendObject(obj, ldm_ids[matchedObj[0]])
                    continue
            # we are detecting a new object
            # newID = self.LDM_ids.pop()
            newID = obj.id
            if obj.id == 1:
                print('newID:', newID)
            print(f'New ID: {newID}')
            self.LDM[newID] = newLDMentry(obj, newID, connected=False, onSight=True)
            self.LDM[newID].kalman_filter = PO_kalman_filter()
            self.LDM[newID].kalman_filter.init_step(obj.xPosition,
                                                    obj.yPosition,
                                                    vx=self.cav.vehicle.get_velocity().x,
                                                    vy=self.cav.vehicle.get_velocity().y,
                                                    ax=self.cav.vehicle.get_acceleration().x,
                                                    ay=self.cav.vehicle.get_acceleration().y)

        # Delete old perceptions
        if self.cav.time > 2.0:
            T = self.cav.time - 2.0
            old_ids = [ID for ID, LDMobj in self.LDM.items() if LDMobj.getLatestPoint().timestamp <= T]
            for ID in old_ids:
                del self.LDM[ID]
                self.LDM_ids.add(ID)

        if self.cav.time > 2.0:
            T = self.cav.time - 0.5
            old_ids = [ID for ID, LDMobj in self.LDM.items() if LDMobj.getLatestPoint().timestamp <= T
                       and (self.cav.vehicle.id not in LDMobj.perceivedBy or not LDMobj.tracked)]
            for ID in old_ids:
                del self.LDM[ID]
                self.LDM_ids.add(ID)

        # Clean possible duplicates
        if len(self.LDM) != 0:
            self.clean_duplicates()

        self.last_update = self.cav.time
        # LDM visualization in lidar view
        showObjects = self.LDM_to_lidarObjects()
        gt = self.cav.perception_manager.getGTobjects()
        #for PO in POS = self.getAllPOs()
        
        if self.cav.perception_manager.lidar:
            while self.cav.perception_manager.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.cav.localizer.get_ego_pos(),
                                  self.cav.perception_manager.lidar.data,
                                  self.cav.perception_manager.lidar.o3d_pointcloud)
            if self.cav.lidar_visualize:
                o3d_visualizer_showLDM(
                    self.o3d_vis,
                    self.cav.perception_manager.count,
                    self.cav.perception_manager.lidar.o3d_pointcloud,
                    showObjects,
                    gt)

    def LDM_to_lidarObjects(self):
        lidarObjects = []
        for ID, LDMobj in self.LDM.items():
            egoPos = self.cav.localizer.get_ego_pos()
            dist = math.sqrt(math.pow((LDMobj.perception.xPosition - egoPos.location.x), 2) +
                             math.pow((LDMobj.perception.yPosition - egoPos.location.y), 2))
            if dist < 100:
                lidarObjects.append(LDMobj)  # return last sample of each object in LDM
        return {'vehicles': lidarObjects}

    def clean_duplicates(self):
        objects = [obj.perception for obj in self.LDM.values()]
        print('CLEAN DUPLICATES')
        IoU_map, new, matched, ldm_ids = self.match_LDM(objects)
        indices_to_delete = []
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if IoU_map[i][j] > 0 and not self.LDM[ldm_ids[j]].connected:
                    indices_to_delete.append(j)
        indices_to_delete = list(set(indices_to_delete))
        print(indices_to_delete)

        for i in indices_to_delete:
            del self.LDM[ldm_ids[i]]
            self.LDM_ids.add(ldm_ids[i])

    def appendObject(self, obj, id):
        if self.cav.vehicle.id not in self.LDM[id].perceivedBy:
            self.LDM[id].perceivedBy.append(self.cav.vehicle.id)
        if obj.timestamp < self.LDM[id].getLatestPoint().timestamp:
            print(f'OLD TIMESTAMP for {obj.id}')
            return

        self.LDM[id].onSight = True
        # Compute the estimated heading angle
        obj.heading = obj.yaw
        if self.LDM[id].connected is True:
            # If this entry is of a connected vehicle
            # obj.connected = True  # We do this to take the width and length from CAM always
            print("CONNECTED VEH WITH ID: ", id)
            obj.width = self.LDM[id].perception.width
            obj.length = self.LDM[id].perception.length
            obj.yaw = self.LDM[id].perception.yaw
            obj.heading = self.LDM[id].perception.heading
        else:
            width_max = obj.width
            length_max = obj.length
            for prev_obj in self.LDM[id].pathHistory:
                if prev_obj.width > width_max <= 2.1:
                    width_max = prev_obj.width
                if length_max < prev_obj.length:
                    length_max = prev_obj.length
            obj.width = width_max
            obj.length = length_max
            yaw_list = [obj.yaw]
            for prev_obj in self.LDM[id].pathHistory:
                yaw_list.append(prev_obj.yaw)
            obj.yaw = np.mean(yaw_list)

        obj.o3d_bbx, obj.line_set = LDMobj_to_o3d_bbx(self.cav, obj)

        x, y, vx, vy, ax, ay = self.LDM[id].kalman_filter.update(obj.xPosition, obj.yPosition, self.cav.get_time_ms())
        obj.xPosition = x
        obj.yPosition = y
        obj.xSpeed = vx
        obj.ySpeed = vy
        obj.xacc = ax
        obj.yacc = ay
        # obj.confidence = self.computeGT_accuracy(obj, id)
        self.LDM[id].insertPerception(obj)


    def computeGT_accuracy(self, object, id):
        # compute the IoU between the ground truth and the object
        IoU = 0.0
        world = self.cav.map_manager.world
        # create dictionary with all the vehicles in the world
        vehicle_list = {}
        for actor in world.get_actors().filter("*vehicle*"):
            id_x = actor.id
            vehicle_list[id_x] = actor

        if id in vehicle_list:
            gt = vehicle_list[id]
            gt_bbx, gt_line_set = get_o3d_bbx(self.cav, gt.get_location().x, gt.get_location().y, gt.bounding_box.extent.x * 2,
                                              gt.bounding_box.extent.y * 2, gt.get_transform().rotation.yaw)

            iou = compute_IoU(gt_bbx, object.o3d_bbx)
            try:
                iou = compute_IoU_lineSet(gt_line_set, object.line_set)
            except RuntimeError as e:
                # print("Unable to compute the oriented bounding box:", e)
                pass
            if iou > 0.0:
                IoU = iou
        return IoU

    def cleanDuplicates(self):
        # simpleLDM = self.getLDM()
        duplicates = []
        for ID, LDMobj in self.LDM.items():
            matchedId = matchLDMobject(self, LDMobj)
            if matchedId != -1:
                if LDMobj.PLU is True and self.LDM[matchedId].PLU is False:
                    duplicates.append(matchedId)
                elif LDMobj.connected is False and self.LDM[matchedId].connected:
                    duplicates.append(matchedId)
                elif LDMobj.connected is False and LDMobj.timestamp > self.LDM[matchedId].timestamp:
                    duplicates.append(matchedId)
                elif LDMobj.connected is False and \
                        (LDMobj.width + LDMobj.length) > (self.LDM[matchedId].width + self.LDM[matchedId].length):
                    duplicates.append(matchedId)
                elif LDMobj.connected is True:
                    duplicates.append(matchedId)
                else:
                    duplicates.append(ID)
        deleted = []
        for ID in duplicates:
            if ID not in deleted:
                del self.LDM[ID]
                deleted.append(ID)  # In case we have duplicates in the 'duplicates' list

    def getCPM(self):
        t_map = []
        cpm = {}
        for ID, entry in self.LDM.items():
            t_map.append((entry.getLatestPoint().timestamp, ID))
        sorted_list = sorted(t_map, key=lambda x: x[0], reverse=True)

        for obj in sorted_list[:10]:
            cpm[obj[1]] = self.LDM[obj[1]]
        return cpm

    def getAllPOs(self):
        POs = []
        for ID, LDMobj in self.LDM.items():
            if not LDMobj.connected:
                POs.append(LDMobj)
        return POs

    def CAMfusion(self, CAMobject):
        CAMobject.connected = True
        CAMobject.o3d_bbx, CAMobject.line_set = get_o3d_bbx(self.cav,
                                                            CAMobject.xPosition,
                                                            CAMobject.yPosition,
                                                            CAMobject.width,
                                                            CAMobject.length,
                                                            CAMobject.yaw)
        if CAMobject.id in self.LDM:
            # If this is not the first CAM from this vehicle
            self.LDM[CAMobject.id].kalman_filter.predict(self.cav.get_time_ms())
            x, y, vx, vy, ax, ay = self.LDM[CAMobject.id].kalman_filter.update(CAMobject.xPosition, CAMobject.yPosition,
                                                                               self.cav.get_time_ms())
            # print('KFupdate: ', "x: ", x, ",y: ", y, ",vx: ", vx, ",vy: ", vy, ",ax: ", ax, ",ay: ", ay)
            CAMobject.xPosition = x
            CAMobject.yPosition = y
            # We trust the CAM speed
            # CAMobject.xSpeed = vx
            # CAMobject.ySpeed = vy
            CAMobject.xacc = ax
            CAMobject.yacc = ay
            self.LDM[CAMobject.id].insertPerception(CAMobject)
            self.LDM[CAMobject.id].connected = True
        else:
            # If this is the first CAM, check if we are already perceiving it
            IoU_map, new, matched, ldm_ids = self.match_LDM([CAMobject])
            if IoU_map is not None:
                if IoU_map[matched[0], new[0]] >= 0:
                    # If we are perceiving this object, delete the entry as PO
                    del self.LDM[ldm_ids[matched[0]]]
                    self.LDM_ids.add(ldm_ids[matched[0]])
            # Create new entry
            if CAMobject.id in self.cav.LDM_ids:
                self.cav.LDM_ids.remove(CAMobject.id)
            self.LDM[CAMobject.id] = newLDMentry(CAMobject, CAMobject.id, connected=True, onSight=True)
            self.LDM[CAMobject.id].kalman_filter = PO_kalman_filter()
            self.LDM[CAMobject.id].kalman_filter.init_step(CAMobject.xPosition,
                                                           CAMobject.yPosition,
                                                           CAMobject.xSpeed,
                                                           CAMobject.ySpeed)
        return CAMobject.id

    def CPMfusion(self, object_list, fromID):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        post_list = []
        post_list = object_list # TODO: solve ms-van3t api resulting in changing POids from same cav
        # if fromID in self.recvCPMmap:
        #     for PO in object_list:
        #         if PO.id in self.recvCPMmap[fromID]:
        #             if self.recvCPMmap[fromID][PO.id] in self.LDM:
        #                 self.append_CPM_object(PO, self.recvCPMmap[fromID][PO.id], fromID)
        #                 continue
        #         post_list.append(PO)
        # else:
        #     self.recvCPMmap[fromID] = {}
        #     post_list = object_list

        # Try to match CPM objects with LDM ones
        # If we match an object, we perform fusion averaging the bbx
        # If can't match the object we append it to the LDM as a new object
        print (f'Post list: {post_list}')
        for CPMobj in post_list:
            
            if self.last_update > CPMobj.timestamp:
                diff = self.last_update - CPMobj.timestamp
                # If it's an old perception, we need to predict its current position
                CPMobj.xPosition += CPMobj.xSpeed * diff + 0.5 * CPMobj.xacc * (diff ** 2)
                CPMobj.yPosition += CPMobj.ySpeed * diff + 0.5 * CPMobj.yacc * (diff ** 2)
                CPMobj.o3d_bbx, CPMobj.line_set = get_o3d_bbx(self.cav, CPMobj.xPosition,
                                                              CPMobj.yPosition,
                                                              CPMobj.width,
                                                              CPMobj.length,
                                                              CPMobj.yaw)

        IoU_map, new, matched, ldm_ids = self.match_LDM(post_list)

        for j in range(len(post_list)):
            CPMobj = post_list[j]
            print(f'CPMobj with {CPMobj.id} is here')
            # Compute bbx from cav's POV because we already converted values
            CPMobj.o3d_bbx, CPMobj.line_set = get_o3d_bbx(self.cav, CPMobj.xPosition,
                                                          CPMobj.yPosition,
                                                          CPMobj.width,
                                                          CPMobj.length,
                                                          CPMobj.yaw)
            if IoU_map is not None:
                matchedObj = matched[np.where(new == j)[0]]
                if IoU_map[matchedObj, j] >= 0:
                    self.append_CPM_object(CPMobj, ldm_ids[matchedObj[0]], fromID)
                    # self.recvCPMmap[fromID][CPMobj.id] = ldm_ids[matchedObj[0]]
                    continue
            dist = math.sqrt(
                math.pow((CPMobj.xPosition - ego_pos.location.x), 2) + math.pow(
                    (CPMobj.yPosition - ego_pos.location.y), 2))
            if dist < 3:
                #print(f'CPM OBJ {CPMobj.id} less than 3 distance')
                continue
            # newID = self.LDM_ids.pop() # TODO: solve ms-van3t api resulting in changing POids from same cav
            newID = CPMobj.id
            self.LDM[newID] = newLDMentry(CPMobj, newID, connected=False, onSight=False)
            print(f'CPM OBJ {CPMobj.id} HAS CPM FLAG TRUE')
            self.LDM[newID].CPM = True
            self.LDM[newID].perceivedBy.append(fromID)
            self.LDM[newID].kalman_filter = PO_kalman_filter()
            self.LDM[newID].kalman_filter.init_step(CPMobj.xPosition,
                                                    CPMobj.yPosition,
                                                    CPMobj.xSpeed,
                                                    CPMobj.ySpeed,
                                                    CPMobj.xacc,
                                                    CPMobj.yacc)
            # self.recvCPMmap[fromID][CPMobj.id] = newID

    def append_CPM_object(self, CPMobj, id, fromID):
        if fromID not in self.LDM[id].perceivedBy:
            self.LDM[id].perceivedBy.append(fromID)
        if CPMobj.timestamp < self.LDM[id].getLatestPoint().timestamp - 100:  # Consider objects up to 100ms old
            print(f'CPMobj with id {CPMobj} is old')
            return
        
        newLDMobj = Perception(CPMobj.xPosition,
                               CPMobj.yPosition,
                               CPMobj.width,
                               CPMobj.length,
                               CPMobj.timestamp,
                               CPMobj.confidence)
        newLDMobj.yaw = CPMobj.yaw
        newLDMobj.heading = CPMobj.heading
        # If the object is also perceived locally
        if self.cav.vehicle.id in self.LDM[id].perceivedBy and self.LDM[id].onSight:
            # Compute weights depending on the POage and confidence (~distance from detecting vehicle)
            LDMobj = self.LDM[id].perception
            LDMobj_age = 100 - (self.cav.get_time() - LDMobj.timestamp)
            CPMobj_age = 100 - (self.cav.get_time() - CPMobj.timestamp)
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

            newLDMobj.xPosition = (LDMobj.xPosition * weightCPM + CPMobj.xPosition * weightLDM) / (
                    weightLDM + weightCPM)
            newLDMobj.yPosition = (LDMobj.yPosition * weightCPM + CPMobj.yPosition * weightLDM) / (
                    weightLDM + weightCPM)
            newLDMobj.xSpeed = (CPMobj.xSpeed * weightCPM + LDMobj.xSpeed * weightLDM) / (weightLDM + weightCPM)
            newLDMobj.ySpeed = (CPMobj.ySpeed * weightCPM + LDMobj.ySpeed * weightLDM) / (weightLDM + weightCPM)
            newLDMobj.width = (CPMobj.width * weightCPM + LDMobj.width * weightLDM) / (weightLDM + weightCPM)
            newLDMobj.length = (CPMobj.length * weightCPM + LDMobj.length * weightLDM) / (weightLDM + weightCPM)
            newLDMobj.heading = (CPMobj.heading * weightCPM + LDMobj.heading * weightLDM) / (weightLDM + weightCPM)
            newLDMobj.yaw = (CPMobj.yaw * weightCPM + LDMobj.yaw * weightLDM) / (weightLDM + weightCPM)
            # newLDMobj.confidence = (CPMobj.confidence + LDMobj.confidence) / 2
            newLDMobj.timestamp = self.cav.get_time()
            # self.LDM[id].onSight = True
        else:
            self.LDM[id].onSight = False

        x, y, vx, vy, ax, ay = self.LDM[id].kalman_filter.update(newLDMobj.xPosition, newLDMobj.yPosition,
                                                                 self.cav.get_time_ms())
        # print('KFupdate: ', "x: ", x, ",y: ", y, ",vx: ", vx, ",vy: ", vy, ",ax: ", ax, ",ay: ", ay)
        newLDMobj.xPosition = x
        newLDMobj.yPosition = y
        newLDMobj.xSpeed = vx
        newLDMobj.ySpeed = vy
        newLDMobj.xacc = ax
        newLDMobj.yacc = ay

        newLDMobj.o3d_bbx, newLDMobj.line_set = get_o3d_bbx(self.cav,
                                                            newLDMobj.xPosition,
                                                            newLDMobj.yPosition,
                                                            newLDMobj.width,
                                                            newLDMobj.length,
                                                            newLDMobj.yaw)
        # cav.LDM[id].kalman_filter.update(newLDMobj.xPosition, newLDMobj.yPosition, newLDMobj.width, newLDMobj.length)
        print(f'CPMobj with id {CPMobj} in appendCMPobj')
        self.LDM[id].insertPerception(newLDMobj)
        self.LDM[id].CPM = True

    def getLDM_tracked(self):
        tracked = {}
        for ID, LDMobj in self.LDM.items():
            if LDMobj.tracked:
                tracked[ID] = LDMobj
        return tracked

    def getLDM_perceptions(self):
        perceptions = {}
        for ID, LDMobj in self.LDM.items():
            perceptions[ID] = LDMobj.perception
        return perceptions

    def get_LDM_size(self):
        tracked = 0
        for ID, LDMobj in self.LDM.items():
            if LDMobj.tracked:
                tracked += 1
        return tracked

    def LDM2OpencdaObj(self, trafficLights):
        LDM = self.getLDM_tracked()
        retObjects = []
        for ID, LDMObject in LDM.items():
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
