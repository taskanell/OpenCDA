from opencda.customize.v2x.aux import LDMObject
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
import math
import carla
import time
import csv
import random
from opencda.customize.v2x.LDMutils import CPMfusion


class CPservice(object):
    def __init__(
            self,
            cav,
            V2Xagent):
        self.cav = cav
        self.V2Xagent = V2Xagent
        self.cpm_sent = 0
        self.last_cpm = 0

    def checkCPMconditions(self):
        # TODO: implement the correct redundancy mitigation algorithm
        if (self.cav.time * 1000) - self.last_cpm > 100:
            return True
        else:
            return False

    def generateCPM(self):
        ego_pos, ego_spd, objects = self.cav.getInfo()
        if self.cav.pldm and self.cav.PLDM is not None:
            LDM = self.cav.PLDM.getCPM()
        else:
            LDM = self.cav.getCPM()

        ego_spd = ego_spd / 3.6  # km/h to m/s

        POs = []
        nPOs = 0

        IDs = random.sample(range(1, 256), 10)
        for carlaID, LDMobj in LDM.items():
            if LDMobj.detected is False or LDMobj.onSight is False:
                continue
            if LDMobj.getLatestPoint().timestamp < self.cav.time - 1.0:
                continue

            dx = (LDMobj.perception.xPosition - ego_pos.location.x)
            dy = (LDMobj.perception.yPosition - ego_pos.location.y)
            dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

            relAngle = math.atan2(dy, dx) - math.radians(ego_pos.rotation.yaw)
            xDist = dist * math.cos(relAngle)
            yDist = dist * math.sin(relAngle)

            dxSpeed = (LDMobj.perception.xSpeed - ego_spd * math.cos(math.radians(ego_pos.rotation.yaw)))
            dySpeed = (LDMobj.perception.ySpeed - ego_spd * math.sin(math.radians(ego_pos.rotation.yaw)))
            dSpeed = math.sqrt(math.pow(dxSpeed, 2) + math.pow(dySpeed, 2))
            relAngle = math.atan2(dySpeed, dxSpeed) - math.radians(ego_pos.rotation.yaw)
            xSpeed = dSpeed * math.cos(relAngle)
            ySpeed = dSpeed * math.sin(relAngle)

            # For debugging
            POs.append({'ObjectID': int(IDs[nPOs]),
                        'Heading': int(LDMobj.perception.heading * 10),  # In degrees/10
                        'xSpeed': int(xSpeed * 100),  # Centimeters per second
                        'ySpeed': int(ySpeed * 100),  # Centimeters per second
                        'vehicleWidth': int(LDMobj.perception.width * 10),  # In meters/10
                        'vehicleLength': int(LDMobj.perception.length * 10),  # In meters/10
                        'xDistance': int(xDist * 100),  # Centimeters
                        'yDistance': int(yDist * 100),  # Centimeters
                        'confidence': int((100 - dist) if dist < 100 else 0),
                        'timestamp': int(LDMobj.getLatestPoint().timestamp * 1000)})
            nPOs = nPOs + 1
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

        CPM = {'ETSItype': 'CPM',
               'stationID': self.cav.vehicle.id,
               'numberOfPOs': nPOs,
               'perceivedObjects': POs,
               'referencePosition': referencePosition,
               'stationData': stationData}
        # print(CPM)
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
                dist = math.sqrt(
                    math.pow(float(CPMobj['xDistance']) / 100, 2) + math.pow(float(CPMobj['yDistance']) / 100, 2))
                relAngle = math.atan2(float(CPMobj['yDistance']) / 100, float(CPMobj['xDistance']) / 100)
                absAngle = relAngle + math.radians(carlaTransform.rotation.yaw)
                xPos = dist * math.cos(absAngle) + carlaTransform.location.x
                yPos = dist * math.sin(absAngle) + carlaTransform.location.y

                dSpeed = math.sqrt(
                    math.pow(float(CPMobj['xSpeed']) / 100, 2) + math.pow(float(CPMobj['ySpeed']) / 100, 2))
                relAngle = math.atan2(float(CPMobj['ySpeed']) / 100, float(CPMobj['xSpeed']) / 100)
                absAngle = relAngle + math.radians(
                    carlaTransform.rotation.yaw)  # This absolute value is actually the heading of the object
                xSpeed = dSpeed * math.cos(absAngle) + (float(stationData['speed']) / 100) * math.cos(
                    math.radians(carlaTransform.rotation.yaw))
                ySpeed = dSpeed * math.sin(absAngle) + (float(stationData['speed']) / 100) * math.sin(
                    math.radians(carlaTransform.rotation.yaw))

                if CPMobj['vehicleWidth'] == 0 or CPMobj['vehicleLength'] == 0:
                    continue
                # CPM object converted to LDM format
                newPO = LDMObject(CPMobj['ObjectID'],
                                  xPos,
                                  yPos,
                                  float(CPMobj['vehicleWidth']) / 10,
                                  float(CPMobj['vehicleLength']) / 10,
                                  float(CPMobj['timestamp']) / 1000,
                                  float(CPMobj['confidence']))
                newPO.xSpeed = xSpeed
                newPO.ySpeed = ySpeed
                # print('[CPM] ' + str(newPO.id) + ' speed: ' + str(newPO.xSpeed) + ',' + str(newPO.ySpeed))
                newPO.heading = math.degrees(absAngle)
                newPOs.append(newPO)
            t_parse += time.time_ns() / 1000 - init_t
            init_t = time.time_ns() / 1000
            self.cav.ldm_mutex.acquire()
            CPMfusion(self.cav, newPOs, CPM['stationID'])
            self.cav.ldm_mutex.release()
            t_fusion += time.time_ns() / 1000 - init_t

        return t_parse, t_fusion
