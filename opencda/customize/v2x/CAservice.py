import carla
import math
import numpy as np
import weakref
from threading import Thread
from threading import Event
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform

from opencda.customize.v2x.aux import Perception


class CAservice(object):
    def __init__(
            self,
            cav,
            V2Xagent):

        self.cav = cav
        self.V2Xagent = V2Xagent
        self.prev_heading = -1
        self.prev_speed = -1
        self.prev_distance = -1
        self.cam_sent = 0
        self.next_cam = 0
        self.lastCamGen = 0
        self.max_CAM_T = 100

    def checkCAMconditions(self):
        now = self.cav.get_time_ms()
        condition_verified = False
        dyn_cond_verified = False

        if self.prev_heading == -1 or self.prev_speed == -1 or self.prev_distance == -1:
            return True

        if now - self.lastCamGen < 100:
            return False

        head_diff = self.cav.localizer.get_ego_pos().rotation.yaw - self.prev_heading
        head_diff += -360.0 if head_diff > 180.0 else (360.0 if head_diff < -180.0 else 0.0)
        if head_diff > 4.0 or head_diff < -4.0:
            return True

        pos_diff = self.cav.localizer.get_ego_pos().location.distance(self.prev_distance)
        if pos_diff > 4.0 or pos_diff < -4.0:
            return True

        speed_diff = (self.cav.localizer.get_ego_spd() - self.prev_speed) * 3.6
        if speed_diff > 0.5 or speed_diff < -0.5:
            return True

        if not condition_verified and (now - self.lastCamGen >= self.max_CAM_T):
            return True

        return False

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
        carlaTransform = fromTransform
        newCV = Perception(carlaTransform.location.x,
                           carlaTransform.location.y,
                           float(highFreqContainer['vehicleWidth']) / 10.0,
                           float(highFreqContainer['vehicleLength']) / 10.0,
                           CAM['timestamp'] / 1000,
                           100,
                           float(highFreqContainer['speed']) / 100 * math.cos(
                               math.radians(carlaTransform.rotation.yaw)),
                           float(highFreqContainer['speed']) / 100 * math.sin(
                               math.radians(carlaTransform.rotation.yaw)),
                           carlaTransform.rotation.yaw,
                           ID=CAM['stationID'])
        newCV.yaw = carlaTransform.rotation.yaw
        # print('Vehicle ' + str(self.cav.vehicle.id) + ' received CAM from vehicle ' + str(CAM['stationID']))
        self.cav.ldm_mutex.acquire()
        # ldm_id = CAMfusion(self.cav, newCV)
        ldm_id = self.cav.LDM.CAMfusion(newCV)
        self.cav.ldm_mutex.release()
        if CAM['isJoinable'] is True and self.V2Xagent.pcService is not None:
            self.V2Xagent.pcService.updateJoinableList(ldm_id)
        return True

    def generateCAM(self):
        referencePosition = {
            'altitude': self.cav.localizer.get_ego_geo_pos().altitude,
            'longitude': int(self.cav.localizer.get_ego_geo_pos().longitude * 10000000),
            'latitude': int(self.cav.localizer.get_ego_geo_pos().latitude * 10000000),
        }
        highFreqContainer = {
            'heading': int(self.cav.localizer.get_ego_pos().rotation.yaw * 10),
            'speed': int(self.cav.localizer.get_ego_spd() * 100 / 3.6),
            'vehicleLength': int(self.cav.vehicle.bounding_box.extent.x * 20),
            'vehicleWidth': int(self.cav.vehicle.bounding_box.extent.y * 20) if self.cav.vehicle.bounding_box.extent.y > 0 else 1,
            'acceleration': int(self.cav.localizer.get_ego_acc() * 10) if abs(self.cav.localizer.get_ego_acc() * 10) < 160 else 161,
            'yawRate': int(self.cav.localizer.get_ego_pos().rotation.yaw * 10)
        }
        cam = {
            'type': 'CAM',
            'stationID': int(self.cav.vehicle.id),
            'timestamp': int((self.cav.time * 1000)) % 65536,
            'referencePosition': referencePosition,
            'highFrequencyContainer': highFreqContainer,
        }
        if self.V2Xagent.pcService is not None:
            cam['isJoinable'] = True if self.V2Xagent.pcService.getIsJoinable() else False
        else:
            cam['isJoinable'] = False

        # print('Vehicle ' + str(self.cav.vehicle.id) + ' sent CAM')
        self.prev_heading = self.cav.localizer.get_ego_pos().rotation.yaw
        self.prev_speed = self.cav.localizer.get_ego_spd()
        self.prev_distance = self.cav.localizer.get_ego_pos().location
        self.lastCamGen = self.cav.time * 1000
        self.cam_sent = 0
        return cam
