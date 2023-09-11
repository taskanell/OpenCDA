import carla
import math
import numpy as np
import weakref
from threading import Thread
from threading import Event
from proton import Message, Url
from proton.handlers import MessagingHandler
from proton.reactor import Container

from opencda.customize.platooning.states import FSM
from collections import deque


class PCservice(object):
    def __init__(
            self,
            cav,
            V2Xagent):
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.pcm_sent = 0
        self.last_pcm = 0
        self.isJoinable = False
        self.joinableList = set()

        # whether leader in a platoon
        self.leader = False
        self.PCMap = {}  # Map containing the last 10 PCM received from each PM
        self.platooning_id = None
        self.leader_id = None
        self.platoon_position = None
        self.status = FSM.SEARCHING

        self.platoon_list = {}
        self.maximum_capacity = 10

        self.destination = None
        self.center_loc = None

        # this is used to control platooning speed during joining
        self.leader_target_speed = 0
        self.origin_leader_target_speed = 0
        self.recover_speed_counter = 0

        # the platoon in the black list won't be considered again
        self.platooning_blacklist = []

        # used to label the front and rear vehicle position
        self.front_vehicle = None
        self.front_front_vehicle = None
        self.rear_vehicle = None

        self.join_req_counter = 0
        self.last_join_req = 0
        self.requestTO = None

        self.join_resp_counter = 0
        self.last_join_resp = 0
        self.responseTO = 0

        self.last_tx_pcm = 0
        self.last_rx_pcm = 0

        self.leave_flag = False
        self.leave_pcm_counter = 0

        self.leave_counter = 0
        self.last_leave_msg = 0

    def getIsJoinable(self):
        return self.isJoinable

    def updateJoinableList(self, id):
        self.joinableList.add(id)

    def getPlatoonStatus(self):
        return self.status

    def checkJoinConditions(self):
        """
        Check conditions for joining a platoon.
        For now this means to check if we are behind a joinable vehicle.
        Returns
        -------

        """
        # TODO: check which of the vehicles in front is closer
        self.cav.ldm_mutex.acquire()
        ldm = self.cav.get_context()
        self.cav.ldm_mutex.release()
        ego_pos, ego_spd, objects = self.cav.getInfo()
        best_id = None
        furthest = 0
        for id in self.joinableList:
            if id in ldm:
                # Calculate vector from obj1 to obj2
                diff = [ldm[id].xPosition - ego_pos.location.x, ldm[id].yPosition - ego_pos.location.y]
                magnitude = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
                normalized = [diff[0] / magnitude, diff[1] / magnitude]
                heading_vector = [math.cos(math.radians(ego_pos.rotation.yaw)),
                                  math.sin(math.radians(ego_pos.rotation.yaw))]
                # Calculate dot product
                dot_product = normalized[0] * heading_vector[0] + normalized[1] * heading_vector[1]
                if dot_product > 0:
                    # ldm[id] is in front
                    if magnitude > furthest:
                        best_id = id
                        furthest = magnitude

        return best_id

    def run_step(self):
        if self.status == FSM.SEARCHING:
            self.isJoinable = True
            if self.joinableList:
                # If CAMs with the isJoinable flag have been received, check if any of those vehicles are in front
                id = self.checkJoinConditions()
                if id:
                    # If vehicle id is in front, trigger JOIN request
                    self.requestTO = id
                    self.status = FSM.JOIN_REQUEST
                    self.join_req_counter = 0

        if self.status == FSM.JOIN_REQUEST:
            if (self.last_join_req + 100 <= self.cav.get_time_ms()) and (self.join_req_counter < 10):
                self.sendPMM('JOIN_REQ')
            elif self.join_req_counter == 10:
                self.status = FSM.SEARCHING
                self.requestTO = None

        if self.status == FSM.JOIN_RESPONSE:
            # self.leader = True
            if (self.last_join_resp + 100 <= self.cav.get_time_ms()) and (self.join_resp_counter < 10):
                self.sendPMM('JOIN_RESP')
            if self.join_resp_counter >= 10:
                if self.platoon_position == 1:
                    self.status = FSM.LEADING_MODE
                else:
                    self.status = FSM.MAINTINING
                self.isJoinable = False

        if self.status == FSM.BACK_JOINING or self.status == FSM.JOINING_FINISHED:
            if self.last_tx_pcm + 50 <= self.cav.get_time_ms():
                self.sendPCM()

        if self.status == FSM.MAINTINING or self.status == FSM.LEADING_MODE:  # ENSEMBLE'S PLATOON
            if self.last_tx_pcm + 50 <= self.cav.get_time_ms():
                self.sendPCM()
            if self.get_lost_rx_pcm():
                self.status = FSM.LEAVE
            if self.platoon_list:
                if self.platoon_position > max(self.platoon_list.keys()) and self.pcm_sent > 5:
                    self.isJoinable = True

        if self.status == FSM.LEAVE_REQUEST:
            if self.last_tx_pcm + 50 <= self.cav.get_time_ms():
                self.sendPCM(leave=True)
            if self.leave_pcm_counter == 10:
                self.status = FSM.LEAVE

        if self.status == FSM.LEAVE:
            if (self.last_leave_msg + 100 <= self.cav.get_time_ms()) and (self.leave_counter < 10):
                self.sendPMM('LEAVE')
            if self.leave_counter == 10:
                self.status = FSM.SEARCHING

    def get_lost_rx_pcm(self):
        # print(self.PCMap.items())
        for id, pm in self.PCMap.items():
            if pm[(len(self.PCMap[id]) - 1)]['timestamp'] <= (
                    (int(self.cav.get_time_ms()) % 65536) - 500):  # if older than 500ms
                return False
        return False

    def sendPCM(self, leave=False):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        trajectory = self.cav.agent.get_local_planner().get_trajectory()
        trajectory_array = []
        for point in trajectory:
            trajectory_array.append({
                "x": float(point[0].location.x),
                "y": float(point[0].location.y),
                "s": float(point[1])
            })
        referencePosition = {
            'altitude': self.cav.localizer.get_ego_geo_pos().altitude,
            'longitude': int(self.cav.localizer.get_ego_geo_pos().longitude * 10000000),
            'latitude': int(self.cav.localizer.get_ego_geo_pos().latitude * 10000000),
            # Aux fields until I find a way to convert geodesic to cartesian within CARLA
            'carlaX': float(ego_pos.location.x),
            'carlaY': float(ego_pos.location.y)
        }
        longitudinalControlContainer = {
            'grossCombinationVehicleWeight': False,
            'currentLongitudinalAcceleration': False,
            'predictedLongitudinalAcceleration': False,
            'longitudinalSpeed': int(self.cav.localizer.get_ego_spd() * 100 / 3.6),
            'powerToMassRatio': False,
            'brakeCapacity': False,
            'roadInclination': False,
            'referenceSpeed': False,
            'intruderAhead': False,
            'vehicleAhead': False,
        }
        platoonControlContainer = {
            'referencePosition': referencePosition,
            'heading': int(self.cav.localizer.get_ego_pos().rotation.yaw * 10),
            'timestamp': int(self.cav.get_time_ms()) % 65536,
            'sequenceNumber': self.pcm_sent,
            'platoonPosition': self.platoon_position,
            'stationID': int(self.cav.vehicle.id),
            'vehicleLength': int(self.cav.vehicle.bounding_box.extent.x * 20),
            'longitudinalControl': longitudinalControlContainer,
            'aboutToLeave': leave,
        }
        PCM = {
            'type': 'PCM',
            'stationID': int(self.cav.vehicle.id),
            'timestamp': int(self.cav.get_time_ms()) % 65536,
            'platoonControlContainer': platoonControlContainer,
            'trajectory': trajectory_array
        }
        self.last_pcm = self.cav.get_time_ms()
        # self.V2Xagent.AMQPhandler.platoonControl_sender(PCM)
        self.V2Xagent.send_buffer.append(PCM)
        self.V2Xagent.send_event.set()
        self.pcm_sent += 1

    def processPCM(self, pcm):
        # if (self.status == FSM.LEADING_MODE) or (self.status == FSM.MAINTINING) or (
        #         self.status == FSM.JOIN_REQUEST) or (self.status == FSM.BACK_JOINING):
        if self.status != FSM.SEARCHING:
            if pcm['stationID'] in self.PCMap:
                self.PCMap[int(pcm['stationID'])].append(pcm)
            else:
                self.PCMap[int(pcm['stationID'])] = deque([pcm], maxlen=10)

            self.platoon_list[pcm['platoonControlContainer']['platoonPosition']] = pcm['stationID']

            if pcm['platoonControlContainer']['platoonPosition'] == self.platoon_position - 1:
                self.front_vehicle = pcm
            if pcm['platoonControlContainer']['platoonPosition'] == self.platoon_position - 2:
                self.front_vehicle = pcm
            if pcm['platoonControlContainer']['platoonPosition'] == self.platoon_position + 1:
                self.rear_vehicle = pcm
        return True

    def sendPMM(self, type):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        referencePosition = {
            'altitude': self.cav.localizer.get_ego_geo_pos().altitude,
            'longitude': int(self.cav.localizer.get_ego_geo_pos().longitude * 10000000),
            'latitude': int(self.cav.localizer.get_ego_geo_pos().latitude * 10000000),
            # Aux fields until I find a way to convert geodesic to cartesian within CARLA
            'carlaX': float(ego_pos.location.x),
            'carlaY': float(ego_pos.location.y)
        }
        message = None
        if type == 'JOIN_REQ':
            message = self.sendJOINreq()
        if type == 'JOIN_RESP':
            message = self.sendJOINresp()
        if type == 'LEAVE':
            message = self.sendLEAVEmsg()

        PMM = {'type': 'PMM',
               'stationID': int(self.cav.vehicle.id),
               'timestamp': int(self.cav.get_time_ms()) % 65536,
               'referencePosition': referencePosition,
               'heading': int(self.cav.localizer.get_ego_pos().rotation.yaw * 10),
               'messageType': type,
               'message': message
               }
        # self.V2Xagent.AMQPhandler.platoonControl_sender(PMM)
        self.V2Xagent.send_buffer.append(PMM)
        self.V2Xagent.send_event.set()

    def sendJOINreq(self):

        joinRequest = {
            'receiver': self.requestTO,
            'brakeCapacity': False,  # Unavailable
            'powerToMassRatio': False,  # Unavailable
            'platooningLevel': int(0),  # longitudinalOnly
            'vehicleLength': int(self.cav.vehicle.bounding_box.extent.x * 20)
        }

        self.join_req_counter += 1
        self.last_join_req = self.cav.get_time_ms()
        return joinRequest

    def sendJOINresp(self):
        joinResponseStatus = {
            'symmetricKey': False,
            'frequencyChannel': False,
            'platoonId': int(0),  # TODO: find a criteria to generate platoon id
            'maxNofVehiclesInPlatoon': int(self.maximum_capacity),
            'joiningPosition': int(self.platoon_position + 1)
        }
        joinResponse = {
            'respondingTo': self.responseTO,
            'joinResponseStatus': joinResponseStatus
        }

        self.join_resp_counter += 1
        self.last_join_resp = self.cav.get_time_ms()
        return joinResponse

    def sendLEAVEmsg(self):
        leaveRequest = {
            'vehicleId': int(self.cav.vehicle.id),
            'platoonPosition': int(self.platoon_position),
            'reason': 'unavailable'
        }
        self.leave_counter += 1
        return leaveRequest

    def processPMM(self, pmm):
        if pmm['messageType'] == 'JOIN_REQ':
            self.processJOINreq(pmm)
        if pmm['messageType'] == 'JOIN_RESP':
            self.processJOINresp(pmm)
        if pmm['messageType'] == 'LEAVE':
            self.processLEAVE(pmm)
        return True

    def processJOINreq(self, pmm):
        join_req = pmm['message']
        if join_req['receiver'] == int(self.cav.vehicle.id):
            if self.status == FSM.SEARCHING or self.status == FSM.LEADING_MODE:
                self.leader = True
                self.platoon_list[1] = (int(self.cav.vehicle.id))
                self.platoon_list[2] = (int(pmm['stationID']))
                self.platoon_position = 1
                self.responseTO = pmm['stationID']
                self.status = FSM.JOIN_RESPONSE
                self.sendPMM('JOIN_RESP')
            # TODO: consider the case where a JOIN_REQ is received while performing a JOIN
            # e.g. for the case where 2 vehicles want to join, for now we do first req first served
            elif self.status == FSM.MAINTINING:
                self.platoon_list[self.platoon_position+1] = (int(pmm['stationID']))
                self.responseTO = pmm['stationID']
                self.status = FSM.JOIN_RESPONSE
                self.sendPMM('JOIN_RESP')
        return True

    def processJOINresp(self, pmm):
        join_resp = pmm['message']
        status = join_resp['joinResponseStatus']
        if join_resp['respondingTo'] == int(self.cav.vehicle.id):
            if self.status == FSM.JOIN_REQUEST:
                self.status = FSM.BACK_JOINING  # To start the JOINING
                self.platoon_position = status['joiningPosition']
                if status['joiningPosition'] == 'maxNofVehiclesInPlatoon':
                    self.isJoinable = False
                self.platoon_list[status['joiningPosition']-1] = (int(pmm['stationID']))
        return True

    def processLEAVE(self, pmm):
        if self.status == FSM.MAINTINING:
            # If someone in front of the receiving vehicle wants to leave, ego vehicle leaves too
            if self.PCMap[pmm['message']['platoonPosition']] < self.platoon_position:
                self.status = FSM.LEAVE
        if self.status == FSM.LEADING_MODE:
            if self.PCMap[pmm['message']['platoonPosition']] == 2:
                self.status = FSM.LEAVE
            else:
                if pmm['stationID'] in self.PCMap:
                    del self.PCMap[pmm['stationID']]
                if pmm['stationID'] in self.platoon_list:
                    self.platoon_list.remove(pmm['stationID'])

        return True
