from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
import math
import carla
import time
import csv
import random
from opencda.customize.v2x.aux import Perception


class CPservice(object):
    def __init__(
            self,
            cav,
            V2Xagent,
            ObjInclusionConfig=1):
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.cpm_sent = 0
        self.last_cpm = 0
        self.recvCPMmap = {}
        self.lastIncluded = []
        self.objInclusionConfig = ObjInclusionConfig
        self.minPositionChangeThreshold = 4.0  # meters
        self.minGroundSpeedChangeThreshold = 0.5  # m/s
        self.minHeadingChangeThreshold = 4.0  # degrees

    def checkCPMconditions(self):
        # TODO: implement the correct redundancy mitigation algorithm
        CPM = {}
        if (self.cav.time * 1000) - self.last_cpm > 100:
            if self.cav.pldm and self.cav.PLDM is not None:
                LDM = self.cav.PLDM.getCPM()
            else:
                LDM = self.cav.LDM.getCPM()

            if self.objInclusionConfig == 0:
                CPM = LDM
            else:
                for ID, LDMobj in LDM.items():
                    check = (
                            LDMobj.detected and
                            LDMobj.onSight and
                            LDMobj.tracked and
                            LDMobj.getLatestPoint().timestamp >= self.cav.time - 1.0
                    )
                    if not check:
                        continue
                    if ID not in self.lastIncluded or LDMobj.CPM_lastIncluded is None:
                        CPM[ID] = LDMobj
                        continue
                    if self.cav.LDM.LDM[ID].CPM_lastIncluded is not None:
                        dx = (LDMobj.perception.xPosition - self.cav.LDM.LDM[ID].CPM_lastIncluded.xPosition)
                        dy = (LDMobj.perception.yPosition - self.cav.LDM.LDM[ID].CPM_lastIncluded.yPosition)
                        dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                        if dist > self.minPositionChangeThreshold:
                            CPM[ID] = LDMobj
                            continue
                        dvx = (LDMobj.perception.xSpeed - self.cav.LDM.LDM[ID].CPM_lastIncluded.xSpeed)
                        dvy = (LDMobj.perception.ySpeed - self.cav.LDM.LDM[ID].CPM_lastIncluded.ySpeed)
                        dv = math.sqrt(math.pow(dvx, 2) + math.pow(dvy, 2))
                        if dv > self.minGroundSpeedChangeThreshold:
                            CPM[ID] = LDMobj
                            continue
                        dheading = math.degrees(math.atan2(dvy, dvx))
                        if dheading > self.minHeadingChangeThreshold and (math.sqrt(math.pow(LDMobj.perception.xSpeed, 2) + math.pow(LDMobj.perception.ySpeed, 2)) > 1.0):
                            CPM[ID] = LDMobj
                            continue
                        if self.cav.LDM.LDM[ID].CPM_lastIncluded.timestamp < self.cav.time - 1.0:
                            CPM[ID] = LDMobj
                            continue
            return CPM
        else:
            return False

    def generateCPM(self, CPM):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        # if self.cav.pldm and self.cav.PLDM is not None:
        #     CPM = self.cav.PLDM.getCPM()
        # else:
        #     LDM = self.cav.LDM.getCPM()

        ego_spd = ego_spd / 3.6  # km/h to m/s

        POs = []
        nPOs = 0

        # IDs = random.sample(range(1, 256), 10)
        for carlaID, LDMobj in CPM.items():


            dx = (LDMobj.perception.xPosition - ego_pos.location.x)
            dy = (LDMobj.perception.yPosition - ego_pos.location.y)
            dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

            heading = math.degrees(math.atan2(LDMobj.perception.ySpeed, LDMobj.perception.xSpeed))

            # For debugging
            POs.append({'ObjectID': int(LDMobj.id),
                        'Heading': int(heading * 10),  # In degrees/10
                        'xSpeed': int(LDMobj.perception.xSpeed * 100),  # Centimeters per second
                        'ySpeed': int(LDMobj.perception.ySpeed * 100),  # Centimeters per second
                        'xAcceleration': int(LDMobj.perception.xacc * 100),
                        'yAcceleration': int(LDMobj.perception.yacc * 100),
                        'vehicleWidth': int(LDMobj.perception.width * 10),  # In meters/10
                        'vehicleLength': int(LDMobj.perception.length * 10),  # In meters/10
                        'xDistance': int(dx * 100),  # Centimeters
                        'yDistance': int(dy * 100),  # Centimeters
                        'confidence': int((100 - dist) if dist < 100 else 0),
                        'timestamp': int(LDMobj.getLatestPoint().timestamp * 1000)})
            nPOs = nPOs + 1
            if self.cav.pldm and self.cav.PLDM is not None:
                self.cav.PLDM.PLDM[LDMobj.id].CPM_lastIncluded = LDMobj.perception
            else:
                self.cav.LDM.LDM[LDMobj.id].CPM_lastIncluded = LDMobj.perception
            self.lastIncluded.append(LDMobj.id)
            if nPOs == 10:
                break

        referencePosition = {
            'altitude': self.cav.localizer.get_ego_geo_pos().altitude,
            'longitude': int(self.cav.localizer.get_ego_geo_pos().longitude * 10000000),  # 0,1 microdegrees
            'latitude': int(self.cav.localizer.get_ego_geo_pos().latitude * 10000000),  # 0,1 microdegrees
        }

        stationData = {
            'heading': int(ego_pos.rotation.yaw * 10),  # In degrees/10
            'speed': int(ego_spd * 100),  # Centimeters per second
            'vehicleLength': int(self.cav.vehicle.bounding_box.extent.x * 20),  # Centimeters
            'vehicleWidth': int(self.cav.vehicle.bounding_box.extent.y * 20)  # Centimeters
        }

        CPM = {'type': 'CPM',
               'stationID': self.cav.vehicle.id,
               'numberOfPOs': nPOs,
               'perceivedObjects': POs,
               'referencePosition': referencePosition,
               'stationData': stationData}
        # print(CPM, '\n')
        self.last_cpm = self.cav.time * 1000
        return CPM

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
                xPos = carlaTransform.location.x + float(CPMobj['xDistance']) / 100
                yPos = carlaTransform.location.y + float(CPMobj['yDistance']) / 100

                if CPMobj['vehicleWidth'] == 0 or CPMobj['vehicleLength'] == 0:
                    continue
                # CPM object converted to LDM format
                newPO = Perception(xPos,
                                   yPos,
                                   float(CPMobj['vehicleWidth']) / 10,
                                   float(CPMobj['vehicleLength']) / 10,
                                   float(CPMobj['timestamp']) / 1000,
                                   float(CPMobj['confidence']),
                                   ID=CPMobj['ObjectID'])

                newPO.xSpeed = float(CPMobj['xSpeed']) / 100
                newPO.ySpeed = float(CPMobj['ySpeed']) / 100
                newPO.xacc = float(CPMobj['xAcceleration']) / 100
                newPO.yacc = float(CPMobj['yAcceleration']) / 100
                # print('[CPM] ' + str(newPO.id) + ' speed: ' + str(newPO.xSpeed) + ',' + str(newPO.ySpeed))
                newPO.heading = math.degrees(float(CPMobj['Heading']) / 10)
                newPOs.append(newPO)
            t_parse += time.time_ns() / 1000 - init_t
            init_t = time.time_ns() / 1000
            # if CPM['stationID'] in self.recvCPMmap: print("Vehicle ", self.cav.vehicle.id, ":\n"," Previous: ",
            # [x.id for x in self.recvCPMmap[CPM['stationID']]], "\n New: ", [x.id for x in newPOs])
            self.recvCPMmap[CPM['stationID']] = newPOs
            self.cav.ldm_mutex.acquire()
            self.cav.LDM.CPMfusion(newPOs, CPM['stationID'])
            self.cav.ldm_mutex.release()
            t_fusion += time.time_ns() / 1000 - init_t

        return t_parse, t_fusion
