from opencda.customize.v2x.aux import PLDMentry
from opencda.customize.v2x.aux import Perception
from opencda.customize.v2x.PLDM import PLDM
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
import math
import carla
import psutil
import numpy as np
from opencda.customize.v2x.LDMutils import compute_IoU
from opencda.customize.v2x.LDMutils import PO_kalman_filter
from opencda.customize.v2x.LDMutils import get_o3d_bbx
from scipy.optimize import linear_sum_assignment as linear_assignment
from opencda.customize.v2x.aux import newPLDMentry
import time


class PLDMservice(object):
    def __init__(
            self,
            cav,
            V2Xagent):
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.leader = False
        self.leaderSeqNum = 0
        self.recvSeqNum = 0
        self.recv_plu = 0
        self.recv_pmu = {}
        self.last_plu = 0
        self.last_pmu = 0
        self.PMs = []
        self.pldm = PLDM(self.cav, self.V2Xagent, visualize=self.cav.lidar_visualize)
        self.cav.PLDM = self.pldm

    def setLeader(self, leader):
        self.leader = leader
        self.pldm.leader = leader

    def runStep(self):
        if self.leader and (self.cav.time * 1000) - self.last_plu > 100:
            # if len(set(self.recv_pmu.values())) <= 1:
            # Only send PLU if we have received a PMU from all PMs
            self.assignPOs()
            self.generatePLU()
            self.leaderSeqNum += 1

    def assignPOs(self):
        self.optDwell()
        self.balance_resp_pos()

        # For debugging
        resp = self.get_all_PM_resp_state()
        print('PM resp: ')
        for PM, POs in resp:
            print('PM ', PM, ' POs ', [PO.id for PO in POs if PO.tracked])

    def getAssignedPOs(self):
        POs = []
        for ID, PLDMobj in self.pldm.PLDM.items():
            if PLDMobj.detected and PLDMobj.assignedPM == self.cav.vehicle.id:
                POs.append(PLDMobj)
        return POs

    def getAllPOs(self):
        POs = []
        for ID, PLDMobj in self.pldm.PLDM.items():
            if PLDMobj.detected:
                POs.append(PLDMobj)
        return POs

    def optDwell(self):
        allPOs = self.getAllPOs()
        for PO in allPOs:
            POpredX = PO.perception.xPosition + PO.perception.xSpeed  # Predicted position in 1 second
            POpredY = PO.perception.yPosition + PO.perception.ySpeed
            # POpredbbx = self.cav.LDMobj_to_o3d_bbx(PO.perception)
            assocPMs = PO.perceivedBy
            dists = {}
            for assocPM in assocPMs:
                if assocPM in self.pldm.PLDM:
                    PMpredX = self.pldm.PLDM[assocPM].perception.xPosition + self.pldm.PLDM[
                        assocPM].perception.xSpeed  # Predicted position in 1 second
                    PMpredY = self.pldm.PLDM[assocPM].perception.yPosition + self.pldm.PLDM[assocPM].perception.ySpeed
                    # PMpredbbx = self.cav.LDMobj_to_o3d_bbx(self.pldm.PLDM[assocPM].perception)
                    dists[assocPM] = math.sqrt(math.pow((POpredX - PMpredX), 2) + math.pow((POpredY - PMpredY), 2))

            if PO.assignedPM == self.cav.vehicle.id:
                ego_pos, ego_spd, objects = self.cav.getInfo()
                PMpredX = ego_pos.location.x + ego_spd * math.cos(
                    math.radians(ego_pos.rotation.yaw))  # Predicted position in 1 second
                PMpredY = ego_pos.location.y + ego_spd * math.sin(math.radians(ego_pos.rotation.yaw))
                dists[self.cav.vehicle.id] = math.sqrt(
                    math.pow((POpredX - PMpredX), 2) + math.pow((POpredY - PMpredY), 2))

            if dists:
                closest_PM = min(dists, key=dists.get)
                if closest_PM != PO.assignedPM:
                    # Update PO entry in pLDM
                    PO.assignedPM = closest_PM

    def get_all_PM_resp_state(self):
        pm_resps = {}
        pm_resps[self.cav.vehicle.id] = []
        for PM in self.PMs:
            pm_resps[PM] = []

        for ID, PO in self.pldm.PLDM.items():
            if PO.detected:
                pm_resps[PO.assignedPM].append(PO)

        sorted_items = sorted(pm_resps.items(), key=lambda item: len(item[1]), reverse=True)

        pm_resps_ordered = [(key, value) for key, value in sorted_items]


        return pm_resps_ordered

    def balance_resp_pos(self):
        pm_resps = self.get_all_PM_resp_state()

        for PM_most, assignedPOs_most in pm_resps:
            if len(assignedPOs_most) < 2:
                return

            for PM_least, assignedPOs_least in reversed(pm_resps):
                if len(assignedPOs_most) - len(assignedPOs_least) < 2:
                    return

                for PO in assignedPOs_most:
                    if PM_least in PO.perceivedBy:
                        PO.assignedPM = PM_least
                        break

        pm_resps = self.get_all_PM_resp_state()
        return pm_resps

    def generatePLU(self):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        PLDM = self.pldm.PLDM

        ego_spd = ego_spd / 3.6  # km/h to m/s

        POs = []
        PMs = []
        CVs = []

        for carlaID, LDMobj in PLDM.items():
            if LDMobj.detected is False:
                if LDMobj.PM:
                    PMs.append(LDMobj.id)
                else:
                    CVs.append(LDMobj.id)
                continue
            if LDMobj.getLatestPoint().timestamp < self.cav.time - 1.0 or not LDMobj.tracked:
                continue

            dx = (LDMobj.perception.xPosition - ego_pos.location.x)
            dy = (LDMobj.perception.yPosition - ego_pos.location.y)
            dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

            geoPos = self.cav.carla_map.transform_to_geolocation(carla.Location(
                x=LDMobj.perception.xPosition, y=LDMobj.perception.yPosition, z=0))

            referencePosition = {
                'altitude': geoPos.altitude,
                'longitude': int(geoPos.longitude * 10000000),  # 0,1 microdegrees
                'latitude': int(geoPos.latitude * 10000000),  # 0,1 microdegrees
            }
            # speed = math.sqrt(math.pow(LDMobj.perception.xSpeed, 2) + math.pow(LDMobj.perception.ySpeed, 2))

            POs.append({'ObjectID': int(LDMobj.id),
                        'Heading': LDMobj.perception.heading * 10,  # In degrees/10
                        'xSpeed': int(LDMobj.perception.xSpeed * 100),  # Centimeters per second
                        'ySpeed': int(LDMobj.perception.ySpeed * 100),  # Centimeters per second
                        'xAcceleration': int(LDMobj.perception.xacc * 100),
                        'yAcceleration': int(LDMobj.perception.yacc * 100),
                        'vehicleWidth': LDMobj.perception.width * 10,  # In meters/10
                        'vehicleLength': LDMobj.perception.length * 10,  # In meters/10
                        'referencePosition': referencePosition,
                        'confidence': (100 - dist) if dist < 100 else 0,
                        'assocPMs': LDMobj.perceivedBy,
                        'assignedPM': LDMobj.assignedPM,
                        'timestamp': LDMobj.getLatestPoint().timestamp * 1000})

        referencePosition = {
            'altitude': self.cav.localizer.get_ego_geo_pos().altitude,
            'longitude': int(self.cav.localizer.get_ego_geo_pos().longitude * 10000000),  # 0,1 microdegrees
            'latitude': int(self.cav.localizer.get_ego_geo_pos().latitude * 10000000),  # 0,1 microdegrees
        }

        stationData = {
            'heading': int(ego_pos.rotation.yaw * 10),  # In degrees/10
            'speed': int(ego_spd * 100),  # Centimeters per second
            'vehicleLength': int(self.cav.vehicle.bounding_box.extent.x * 20),  # Centimeters
            'vehicleWidth': int(self.cav.vehicle.bounding_box.extent.y * 20),  # Centimeters
            'referencePosition': referencePosition
        }

        PLU = {'type': 'PLU',
               'stationID': self.cav.vehicle.id,
               'timestamp': self.cav.time * 1000,
               'perceivedObjects': POs,
               'stationData': stationData,
               'PMs': PMs,
               'CVs': CVs,
               'seqNum': self.leaderSeqNum}

        self.last_plu = self.cav.time * 1000
        # self.V2Xagent.AMQPhandler.platoonControl_sender(PLU)
        self.V2Xagent.send_buffer.append(PLU)
        self.V2Xagent.send_event.set()
        # print('PLU sent by ', self.cav.vehicle.id, ', tst ', PLU['timestamp'], ', seqNum ', self.leaderSeqNum)

    def generatePMU(self):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        PLDM = self.pldm.PLDM

        ego_spd = ego_spd / 3.6  # km/h to m/s

        assignedPOs = []
        newPOs = []
        perceivedPOs = []

        for carlaID, LDMobj in PLDM.items():
            if not LDMobj.detected or not LDMobj.tracked:
                continue

            if LDMobj.onSight:
                perceivedPOs.append(LDMobj.id)
            if LDMobj.assignedPM == self.cav.vehicle.id or LDMobj.newPO:
                dx = (LDMobj.perception.xPosition - ego_pos.location.x)
                dy = (LDMobj.perception.yPosition - ego_pos.location.y)
                dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

                geoPos = self.cav.carla_map.transform_to_geolocation(carla.Location(
                    x=LDMobj.perception.xPosition, y=LDMobj.perception.yPosition, z=0))

                referencePosition = {
                    'altitude': geoPos.altitude,
                    'longitude': int(geoPos.longitude * 10000000),  # 0,1 microdegrees
                    'latitude': int(geoPos.latitude * 10000000),  # 0,1 microdegrees
                }
                # speed = math.sqrt(math.pow(LDMobj.perception.xSpeed, 2) + math.pow(LDMobj.perception.ySpeed, 2))

                PO = {'ObjectID': int(LDMobj.id),
                      'Heading': LDMobj.perception.heading * 10,  # In degrees/10
                      'xSpeed': int(LDMobj.perception.xSpeed * 100),  # Centimeters per second
                      'ySpeed': int(LDMobj.perception.ySpeed * 100),  # Centimeters per second
                      'xAcceleration': int(LDMobj.perception.xacc * 100),
                      'yAcceleration': int(LDMobj.perception.yacc * 100),
                      'vehicleWidth': LDMobj.perception.width * 10,  # In meters/10
                      'vehicleLength': LDMobj.perception.length * 10,  # In meters/10
                      'referencePosition': referencePosition,
                      'confidence': (100 - dist) if dist < 100 else 0,
                      'assocPMs': LDMobj.perceivedBy,
                      'assignedPM': LDMobj.assignedPM,
                      'timestamp': LDMobj.getLatestPoint().timestamp * 1000}

                if LDMobj.newPO:
                    newPOs.append(PO)
                else:
                    assignedPOs.append(PO)

        PMstate = {'cpuLoad': psutil.cpu_percent(interval=0.1),
                   'perceivedPOs': perceivedPOs,
                   'numberOfAssignedPOs': len(assignedPOs)}

        PMU = {'type': 'PMU',
               'stationID': self.cav.vehicle.id,
               'timestamp': self.cav.get_time_ms(),
               'assignedPOs': assignedPOs,
               'newPOs': newPOs,
               'PMstate': PMstate,
               'seqNum': self.recvSeqNum}

        self.last_pmu = self.cav.time * 1000
        # self.V2Xagent.AMQPhandler.platoonControl_sender(PMU)
        self.V2Xagent.send_buffer.append(PMU)
        self.V2Xagent.send_event.set()
        # print('PMU sent by ', self.cav.vehicle.id, ', tst ', PMU['timestamp'], ', seqNum ', self.recvSeqNum)

    def processPLU(self, PLU):
        if self.leader:
            return
        self.recv_plu += 1
        self.recvSeqNum = PLU['seqNum']
        # print('PLU received by ', self.cav.vehicle.id, ', tst ', PLU['timestamp'], ', seqNum ', PLU['seqNum'],
        #       ',time ', self.cav.get_time_ms())
        t_update = 0
        t_new = 0
        self.cav.pldm_mutex.acquire()
        PLU_ids = []
        if 'perceivedObjects' in PLU:
            init_t = time.time_ns() / 1000
            for PLUobj in PLU['perceivedObjects']:
                PLU_ids.append(PLUobj['ObjectID'])
                if PLUobj['ObjectID'] in self.pldm.PLDM:
                    if self.pldm.PLDM[PLUobj['ObjectID']].newPO:
                        # If it was a newPO but appears in PLU, now is not new
                        self.pldm.PLDM[PLUobj['ObjectID']].newPO = False
                    if PLUobj['assignedPM'] == self.cav.vehicle.id:
                        # If this PM is assigned with this PO, update assignment
                        self.pldm.PLDM[PLUobj['ObjectID']].assignedPM = self.cav.vehicle.id
                        t_update += time.time_ns() / 1000 - init_t
                        continue
                    # If this PM is not assigned with this object, update entry
                    carlaX, carlaY, carlaZ = geo_to_transform(float(PLUobj['referencePosition']['latitude']) / 10000000,
                                                              float(
                                                                  PLUobj['referencePosition']['longitude']) / 10000000,
                                                              PLUobj['referencePosition']['altitude'],
                                                              self.cav.localizer.geo_ref.latitude,
                                                              self.cav.localizer.geo_ref.longitude, 0.0)
                    newPO = Perception(carlaX,
                                       carlaY,
                                       float(PLUobj['vehicleWidth']) / 10,
                                       float(PLUobj['vehicleLength']) / 10,
                                       float(PLUobj['timestamp']) / 1000,
                                       PLUobj['confidence'])
                    newPO.heading = float(PLUobj['Heading']) / 10
                    newPO.xSpeed = float(PLUobj['xSpeed']) / 100
                    newPO.ySpeed = float(PLUobj['ySpeed']) / 100
                    newPO.xacc = float(PLUobj['xAcceleration']) / 100
                    newPO.yacc = float(PLUobj['yAcceleration']) / 100
                    newPO.o3d_bbx = get_o3d_bbx(self.cav, carlaX, carlaY, newPO.width, newPO.length)

                    self.pldm.PLDM[PLUobj['ObjectID']].kalman_filter.predict()
                    x, y, vx, vy, ax, ay = self.pldm.PLDM[PLUobj['ObjectID']].kalman_filter.update(newPO.xPosition,
                                                                                                   newPO.yPosition)
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.xPosition = x
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.yPosition = y
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.xSpeed = vx
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.ySpeed = vy
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.xacc = ax
                    self.pldm.PLDM[PLUobj['ObjectID']].perception.yacc = ay

                    self.pldm.PLDM[PLUobj['ObjectID']].assignedPM = PLUobj['assignedPM']
                    self.pldm.PLDM[PLUobj['ObjectID']].perceivedBy = PLUobj['assocPMs']
                    self.pldm.PLDM[PLUobj['ObjectID']].detected = True
                    self.pldm.PLDM[PLUobj['ObjectID']].PM = False
                    self.pldm.PLDM[PLUobj['ObjectID']].tracked = True
                    self.pldm.PLDM[PLUobj['ObjectID']].insertPerception(newPO)

                else:
                    # If not in local PLDM, create new entry
                    carlaX, carlaY, carlaZ = geo_to_transform(float(PLUobj['referencePosition']['latitude']) / 10000000,
                                                              float(
                                                                  PLUobj['referencePosition']['longitude']) / 10000000,
                                                              PLUobj['referencePosition']['altitude'],
                                                              self.cav.localizer.geo_ref.latitude,
                                                              self.cav.localizer.geo_ref.longitude, 0.0)

                    PO = Perception(carlaX,
                                    carlaY,
                                    float(PLUobj['vehicleWidth']) / 10,
                                    float(PLUobj['vehicleLength']) / 10,
                                    float(PLUobj['timestamp']) / 1000,
                                    PLUobj['confidence'])
                    PO.heading = float(PLUobj['Heading']) / 10
                    PO.xSpeed = float(PLUobj['xSpeed']) / 100
                    PO.ySpeed = float(PLUobj['ySpeed']) / 100
                    PO.xacc = float(PLUobj['xAcceleration']) / 100
                    PO.yacc = float(PLUobj['yAcceleration']) / 100
                    PO.o3d_bbx = get_o3d_bbx(self.cav, carlaX, carlaY, PO.width, PO.length)
                    if PLUobj['ObjectID'] in self.pldm.PLDM_ids:
                        self.pldm.PLDM_ids.remove(PLUobj['ObjectID'])  # To avoid using this ID for a newPO
                    newPO = newPLDMentry(PO, PLUobj['ObjectID'], detected=True, onSight=False)
                    newPO.assignedPM = PLUobj['assignedPM']
                    newPO.perceivedBy = PLUobj['assocPMs']
                    newPO.PM = False
                    newPO.tracked = True
                    newPO.kalman_filter = PO_kalman_filter()
                    newPO.kalman_filter.init_step(PO.xPosition,
                                                  PO.yPosition,
                                                  PO.xSpeed,
                                                  PO.ySpeed,
                                                  PO.xacc,
                                                  PO.yacc)
                    # TODO: see if matching is actually necessary
                    # Check if it doesn't match with a stored perception
                    IoU_map, new, matched, pldm_ids = self.match_PLDMObject([newPO.perception])
                    if IoU_map is not None:
                        if IoU_map[matched[0], new[0]] < 0:
                            # If this is indeed a new object
                            if PLUobj['ObjectID'] in self.pldm.PLDM_ids:
                                self.pldm.PLDM_ids.remove(PLUobj['ObjectID'])  # To avoid using this ID for a newPO
                        else:
                            # If this object is already perceived by PM, delete old entry
                            del self.pldm.PLDM[pldm_ids[matched[0]]]
                            newPO.perceivedBy.append(self.cav.vehicle.id)

                    self.pldm.PLDM[PLUobj['ObjectID']] = newPO
                    # t_new += time.time_ns() / 1000 - init_t
            t_update += time.time_ns() / 1000 - init_t

        init_t = time.time_ns() / 1000
        self.generatePMU()
        t_genPMU = time.time_ns() / 1000 - init_t
        self.cav.pldm_mutex.release()
        # self.cav.pldm_mutex.acquire()
        # extra_ids = []
        # for ID, PLDMobj in self.pldm.PLDM.items():
        #     if ID not in PLU_ids and (PLDMobj.detected and PLDMobj.newPO):
        #         extra_ids.append(ID)
        # for id in extra_ids:
        #     del self.pldm.PLDM[id]
        #     self.pldm.PLDM_ids.add(id)
        # self.cav.pldm_mutex.release()

        return t_update, t_new, t_genPMU

    def processPMU(self, PMU):
        if PMU['stationID'] not in self.PMs:
            self.PMs.append(PMU['stationID'])
        if PMU['stationID'] in self.pldm.PLDM:
            self.pldm.PLDM[PMU['stationID']].PM = True
        if not self.leader:
            return
        # print('PMU received by ', self.cav.vehicle.id, ', tst ', PMU['timestamp'], ', seqNum ', PMU['seqNum'],
        #       ',time ', self.cav.get_time_ms())
        if PMU['stationID'] not in self.recv_pmu:
            self.recv_pmu[PMU['stationID']] = 1
        else:
            self.recv_pmu[PMU['stationID']] += 1
        t_assigned = 0
        t_new = 0
        self.cav.pldm_mutex.acquire()
        if 'assignedPOs' in PMU:
            init_t = time.time_ns() / 1000
            for PMUobj in PMU['assignedPOs']:
                carlaX, carlaY, carlaZ = geo_to_transform(float(PMUobj['referencePosition']['latitude']) / 10000000,
                                                          float(PMUobj['referencePosition']['longitude']) / 10000000,
                                                          PMUobj['referencePosition']['altitude'],
                                                          self.cav.localizer.geo_ref.latitude,
                                                          self.cav.localizer.geo_ref.longitude, 0.0)
                PO = Perception(carlaX,
                                carlaY,
                                float(PMUobj['vehicleWidth']) / 10,
                                float(PMUobj['vehicleLength']) / 10,
                                float(PMUobj['timestamp']) / 1000,
                                PMUobj['confidence'])
                PO.heading = float(PMUobj['Heading']) / 10
                PO.xSpeed = float(PMUobj['xSpeed']) / 100
                PO.ySpeed = float(PMUobj['ySpeed']) / 100
                PO.xacc = float(PMUobj['xAcceleration']) / 100
                PO.yacc = float(PMUobj['yAcceleration']) / 100
                PO.o3d_bbx = get_o3d_bbx(self.cav, carlaX, carlaY, PO.width, PO.length)

                if PMUobj['ObjectID'] in self.pldm.PLDM:
                    self.pldm.PLDM[PMUobj['ObjectID']].kalman_filter.predict()
                    x, y, vx, vy, ax, ay = self.pldm.PLDM[PMUobj['ObjectID']].kalman_filter.update(PO.xPosition,
                                                                                                   PO.yPosition)
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.xPosition = x
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.yPosition = y
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.xSpeed = vx
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.ySpeed = vy
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.xacc = ax
                    self.pldm.PLDM[PMUobj['ObjectID']].perception.yacc = ay
                    self.pldm.PLDM[PMUobj['ObjectID']].assignedPM = PMUobj['assignedPM']
                    self.pldm.PLDM[PMUobj['ObjectID']].perceivedBy = PMUobj['assocPMs']
                    self.pldm.PLDM[PMUobj['ObjectID']].insertPerception(PO)
                else:
                    newPO = newPLDMentry(PO, PMUobj['ObjectID'], detected=True, onSight=False)
                    newPO.assignedPM = PMUobj['assignedPM']
                    newPO.perceivedBy = PMUobj['assocPMs']
                    newPO.kalman_filter = PO_kalman_filter(0.1)
                    newPO.kalman_filter.init_step(PO.xPosition,
                                                  PO.yPosition,
                                                  PO.xSpeed,
                                                  PO.ySpeed,
                                                  PO.xacc,
                                                  PO.yacc)
                    self.pldm.PLDM[PMUobj['ObjectID']] = newPO

                # TODO: test if it works
                if (PMU['timestamp'] - PMUobj['timestamp']) > 200:
                    if self.pldm.PLDM[PMUobj['ObjectID']].onSight:
                        self.pldm.PLDM[PMUobj['ObjectID']].assignedPM = self.cav.vehicle.id
            t_assigned += time.time_ns() / 1000 - init_t
        if 'newPOs' in PMU:
            init_t = time.time_ns() / 1000
            for PMUobj in PMU['newPOs']:
                carlaX, carlaY, carlaZ = geo_to_transform(float(PMUobj['referencePosition']['latitude']) / 10000000,
                                                          float(PMUobj['referencePosition']['longitude']) / 10000000,
                                                          PMUobj['referencePosition']['altitude'],
                                                          self.cav.localizer.geo_ref.latitude,
                                                          self.cav.localizer.geo_ref.longitude, 0.0)

                PO = Perception(carlaX,
                                carlaY,
                                float(PMUobj['vehicleWidth']) / 10,
                                float(PMUobj['vehicleLength']) / 10,
                                float(PMUobj['timestamp']) / 1000,
                                PMUobj['confidence'])
                PO.heading = float(PMUobj['Heading']) / 10
                PO.heading = float(PMUobj['Heading']) / 10
                PO.xSpeed = float(PMUobj['xSpeed']) / 100
                PO.ySpeed = float(PMUobj['ySpeed']) / 100
                PO.xacc = float(PMUobj['xAcceleration']) / 100
                PO.yacc = float(PMUobj['yAcceleration']) / 100
                PO.o3d_bbx = get_o3d_bbx(self.cav, carlaX, carlaY, PO.width, PO.length)

                newPO = newPLDMentry(PO, PMUobj['ObjectID'], detected=True, onSight=False)
                newPO.tracked = True
                newPO.perceivedBy = PMUobj['assocPMs']
                # Check if it doesn't match with a stored perception
                IoU_map, new, matched, pldm_ids = self.match_PLDMObject([newPO.perception])
                if IoU_map is not None:
                    if IoU_map[matched[0], new[0]] < 0:
                        # If this is indeed a new object, assign the object to this PM and create entry
                        newPO.assignedPM = PMU['stationID']
                        newPO.kalman_filter = PO_kalman_filter()
                        newPO.kalman_filter.init_step(PO.xPosition,
                                                      PO.yPosition,
                                                      PO.xSpeed,
                                                      PO.ySpeed,
                                                      PO.xacc,
                                                      PO.yacc)
                        self.pldm.PLDM[PMUobj['ObjectID']] = newPO
                    else:
                        # If this object is already perceived by another PM, add new assoc PM
                        self.pldm.PLDM[pldm_ids[matched[0]]].perceivedBy.append(PMU['stationID'])
            t_new += time.time_ns() / 1000 - init_t
        self.pldm.pmMap[PMU['stationID']] = PMU['PMstate']
        self.cav.pldm_mutex.release()
        return t_new, t_assigned

    def processCPM(self, CPM):
        referencePosition = CPM['referencePosition']
        stationData = CPM['stationData']
        newPOs = []
        t_parse = 0
        t_fusion = 0
        # Compute the CARLA transform with the longitude and latitude values
        carlaX, carlaY, carlaZ = geo_to_transform(float(referencePosition['latitude']) / 10000000,
                                                  float(referencePosition['longitude']) / 10000000,
                                                  referencePosition['altitude'],
                                                  self.cav.localizer.geo_ref.latitude,
                                                  self.cav.localizer.geo_ref.longitude, 0.0)
        carlaTransform = carla.Transform(
            carla.Location(
                x=carlaX, y=carlaY, z=carlaZ),
            carla.Rotation(
                pitch=0, yaw=float(stationData['heading']) / 10, roll=0))
        if 'perceivedObjects' in CPM:
            init_t = time.time_ns() / 1000
            for CPMobj in CPM['perceivedObjects']:
                if CPMobj['ObjectID'] == self.cav.vehicle.id:
                    continue
                # Convert CPM relative values to absolute CARLA values to then match/fusion with LDMobjects
                dist = math.sqrt(
                    math.pow(CPMobj['xDistance'] / 100, 2) + math.pow(CPMobj['yDistance'] / 100, 2))
                relAngle = math.atan2(CPMobj['yDistance'] / 100, CPMobj['xDistance'] / 100)
                absAngle = relAngle + math.radians(carlaTransform.rotation.yaw)
                xPos = dist * math.cos(absAngle) + carlaTransform.location.x
                yPos = dist * math.sin(absAngle) + carlaTransform.location.y

                dSpeed = math.sqrt(math.pow(CPMobj['xSpeed'] / 100, 2) + math.pow(CPMobj['ySpeed'] / 100, 2))
                relAngle = math.atan2(CPMobj['ySpeed'] / 100, CPMobj['xSpeed'] / 100)
                absAngle = relAngle + math.radians(
                    carlaTransform.rotation.yaw)  # This absolute value is actually the heading of the object
                xSpeed = dSpeed * math.cos(absAngle) + (stationData['speed'] / 100) * math.cos(
                    math.radians(carlaTransform.rotation.yaw))
                ySpeed = dSpeed * math.sin(absAngle) + (stationData['speed'] / 100) * math.sin(
                    math.radians(carlaTransform.rotation.yaw))

                # CPM object converted to LDM format
                newPO = Perception(xPos,
                                   yPos,
                                   float(CPMobj['vehicleWidth']) / 10,
                                   float(CPMobj['vehicleLength']) / 10,
                                   float(CPMobj['timestamp']) / 1000,
                                   float(CPMobj['confidence']),
                                   ID=CPMobj['ObjectID'])
                newPO.xSpeed = xSpeed
                newPO.ySpeed = ySpeed
                # print('[CPM] ' + str(newPO.id) + ' speed: ' + str(newPO.xSpeed) + ',' + str(newPO.ySpeed))
                newPO.heading = math.degrees(absAngle)
                newPOs.append(newPO)
            t_parse += time.time_ns() / 1000 - init_t
            init_t = time.time_ns() / 1000
            self.cav.pldm_mutex.acquire()
            self.CPMfusion(newPOs, CPM['stationID'])
            self.cav.pldm_mutex.release()
            t_fusion += time.time_ns() / 1000 - init_t

        return t_parse, t_fusion

    def CPMfusion(self, object_list, fromID):
        if fromID in self.PMs:
            for PO in object_list:
                if PO.id in self.pldm.PLDM:
                    # We only care about POs we are assigned with
                    if self.pldm.PLDM[PO.id].assignedPM == self.cav.vehicle.id:
                        # We don't need to match because ids are (supposed to be) in sync
                        self.append_CPM_object(PO, PO.id, fromID)

    def append_CPM_object(self, CPMobj, id, fromID):
        if fromID not in self.pldm.PLDM[id].perceivedBy:
            self.pldm.PLDM[id].perceivedBy.append(fromID)
        if CPMobj.timestamp < self.pldm.PLDM[id].getLatestPoint().timestamp - 100:  # Consider objects up to 100ms old
            return

        newLDMobj = Perception(CPMobj.xPosition,
                               CPMobj.yPosition,
                               CPMobj.width,
                               CPMobj.length,
                               CPMobj.timestamp,
                               CPMobj.confidence)
        # If the object is also perceived locally
        if self.cav.vehicle.id in self.pldm.PLDM[id].perceivedBy:
            # Compute weights depending on the POage and confidence (~distance from detecting vehicle)
            LDMobj = self.pldm.PLDM[id].perception
            LDMobj_age = LDMobj.timestamp - self.cav.time
            CPMobj_age = CPMobj.timestamp - self.cav.time
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
            # newLDMobj.confidence = (CPMobj.confidence + LDMobj.confidence) / 2
            newLDMobj.timestamp = self.cav.time
            self.pldm.PLDM[id].onSight = True
        else:
            self.pldm.PLDM[id].onSight = False

        newLDMobj.o3d_bbx = get_o3d_bbx(self.cav,
                                        newLDMobj.xPosition,
                                        newLDMobj.yPosition,
                                        newLDMobj.width,
                                        newLDMobj.length)
        # cav.LDM[id].kalman_filter.update(newLDMobj.xPosition, newLDMobj.yPosition, newLDMobj.width, newLDMobj.length)
        self.pldm.PLDM[id].insertPerception(newLDMobj)

    def match_PLDMObject(self, object_list):
        if len(self.pldm.PLDM) != 0:
            IoU_map = np.zeros((len(self.pldm.PLDM), len(object_list)), dtype=np.float32)
            i = 0
            ldm_ids = []
            for ID, PLDMobj in self.pldm.PLDM.items():
                for j in range(len(object_list)):
                    obj = object_list[j]
                    object_list[j].o3d_bbx = self.cav.LDMobj_to_o3d_bbx(obj)
                    LDMpredX = PLDMobj.perception.xPosition
                    LDMpredY = PLDMobj.perception.yPosition
                    # if self.cav.time > PLDMobj.perception.timestamp:
                    #     LDMpredX += (self.cav.time - PLDMobj.perception.timestamp) * PLDMobj.perception.xSpeed
                    #     LDMpredY += (self.cav.time - PLDMobj.perception.timestamp) * PLDMobj.perception.ySpeed
                    #     LDMpredbbx = self.cav.LDMobj_to_o3d_bbx(PLDMobj.perception)
                    LDMpredbbx = get_o3d_bbx(self.cav, LDMpredX, LDMpredY, PLDMobj.perception.width,
                                             PLDMobj.perception.length)

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

    def processCAM(self, CAM):
        referencePosition = CAM['referencePosition']
        highFreqContainer = CAM['highFrequencyContainer']
        # Compute the CARLA transform with the longitude and latitude values
        carlaX, carlaY, carlaZ = geo_to_transform(float(referencePosition['latitude']) / 10000000,
                                                  float(referencePosition['longitude']) / 10000000,
                                                  referencePosition['altitude'],
                                                  self.cav.localizer.geo_ref.latitude,
                                                  self.cav.localizer.geo_ref.longitude, 0.0)

        fromTransform = carla.Transform(
            carla.Location(
                x=carlaX, y=carlaY, z=referencePosition['altitude']), carla.Rotation(
                pitch=0, yaw=float(highFreqContainer['heading']) / 10, roll=0))
        extent = carla.Vector3D(float(highFreqContainer['vehicleLength']) / 10.0,
                                float(highFreqContainer['vehicleWidth']) / 10.0, 0.75)
        carlaTransform = fromTransform
        newCV = Perception(carlaTransform.location.x,
                           carlaTransform.location.y,
                           extent.x,
                           extent.y,
                           CAM['timestamp'] / 1000,
                           100,
                           float(highFreqContainer['speed']) / 100 * math.cos(
                               math.radians(carlaTransform.rotation.yaw)),
                           float(highFreqContainer['speed']) / 100 * math.sin(
                               math.radians(carlaTransform.rotation.yaw)),
                           carlaTransform.rotation.yaw,
                           ID=CAM['stationID'])
        # print('Vehicle ' + str(self.cav.vehicle.id) + ' received CAM from vehicle ' + str(CAM['stationID']))
        self.cav.pldm_mutex.acquire()
        ldm_id = self.CAMfusion(newCV)
        self.cav.pldm_mutex.release()
        if CAM['isJoinable'] is True:
            self.V2Xagent.pcService.updateJoinableList(ldm_id)
        return True

    def CAMfusion(self, CAMobject):
        CAMobject.o3d_bbx = get_o3d_bbx(self.cav,
                                        CAMobject.xPosition,
                                        CAMobject.yPosition,
                                        CAMobject.width,
                                        CAMobject.length)
        if CAMobject.id in self.pldm.PLDM:
            # If this is not the first CAM
            self.pldm.PLDM[CAMobject.id].kalman_filter.predict()
            x, y, vx, vy, ax, ay = self.pldm.PLDM[CAMobject.id].kalman_filter.update(CAMobject.xPosition,
                                                                                     CAMobject.yPosition)
            # print('KFupdate: ', "x: ", x, ",y: ", y, ",vx: ", vx, ",vy: ", vy, ",ax: ", ax, ",ay: ", ay)
            CAMobject.xPosition = x
            CAMobject.yPosition = y
            CAMobject.xSpeed = vx
            CAMobject.ySpeed = vy
            CAMobject.xacc = ax
            CAMobject.yacc = ay
            self.pldm.PLDM[CAMobject.id].insertPerception(CAMobject)
        else:
            # If this is the first CAM, check if we are already perceiving it
            IoU_map, new, matched, ldm_ids = self.match_PLDMObject([CAMobject])
            if IoU_map is not None:
                if IoU_map[matched[0], new[0]] >= 0:
                    # If we are perceiving this object, delete the entry as PO
                    del self.pldm.PLDM[ldm_ids[matched[0]]]
                    self.pldm.PLDM_ids.add(ldm_ids[matched[0]])
            # Create new entry
            if CAMobject.id in self.pldm.PLDM_ids:
                self.pldm.PLDM_ids.remove(CAMobject.id)
            self.pldm.PLDM[CAMobject.id] = newPLDMentry(CAMobject, CAMobject.id, detected=False, onSight=True)
            self.pldm.PLDM[CAMobject.id].PM = True
            self.pldm.PLDM[CAMobject.id].kalman_filter = PO_kalman_filter(0.1)
            self.pldm.PLDM[CAMobject.id].kalman_filter.init_step(CAMobject.xPosition,
                                                                 CAMobject.yPosition,
                                                                 vx=CAMobject.xSpeed,
                                                                 vy=CAMobject.ySpeed)
        return CAMobject.id
