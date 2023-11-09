import carla
import math
import numpy as np
import weakref

from opencda.customize.platooning.states import I_FSM
from collections import deque


class IntruderApp(object):
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
        self.status = I_FSM.IDLE

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
        self.leader_vehicle = None

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
        if self.status == I_FSM.IDLE:
            if self.cav.get_time_ms() > 11500:
                # select random platoon position
                self.platoon_position = np.random.randint(2, self.platoon_list.__len__())
                self.leader_vehicle = self.PCMap[self.platoon_list[1]][-1]
                self.front_vehicle = self.PCMap[self.platoon_list[self.platoon_position - 1]][-1]
                self.rear_vehicle = self.PCMap[self.platoon_list[self.platoon_position + 1]][-1]
                self.status = I_FSM.BEGIN

        # if self.status in (I_FSM.BEGIN, I_FSM.INTRUDE, I_FSM.FINISH):

    def get_lost_rx_pcm(self):
        # print(self.PCMap.items())
        for id, pm in self.PCMap.items():
            if pm[(len(self.PCMap[id]) - 1)]['timestamp'] <= (
                    (int(self.cav.get_time_ms()) % 65536) - 500):  # if older than 500ms
                return False
        return False

    def processPCM(self, pcm):
        # if (self.status == FSM.LEADING_MODE) or (self.status == FSM.MAINTINING) or (
        #         self.status == FSM.JOIN_REQUEST) or (self.status == FSM.BACK_JOINING):
        if pcm['stationID'] in self.PCMap:
            self.PCMap[int(pcm['stationID'])].append(pcm)
        else:
            self.PCMap[int(pcm['stationID'])] = deque([pcm], maxlen=10)

        self.platoon_list[pcm['platoonControlContainer']['platoonPosition']] = pcm['stationID']

        if self.status != I_FSM.IDLE:
            if pcm['platoonControlContainer']['platoonPosition'] == self.platoon_position - 1:
                self.front_vehicle = pcm
            if pcm['platoonControlContainer']['platoonPosition'] == self.platoon_position + 1:
                self.rear_vehicle = pcm
            if pcm['platoonControlContainer']['platoonPosition'] == 1:
                self.leader_vehicle = pcm
        return True


