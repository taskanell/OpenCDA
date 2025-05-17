from opencda.customize.v2x.aux import Perception
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
from opencda.scenario_testing.utils.sim_api import ScenarioManager
from opencda.customize.v2x.LDMutils import get_abs_o3d_bbx
from opencda.customize.v2x.LDMutils import compute_IoU_lineSet
from opencda.customize.v2x.LDMutils import compute_IoU
import numpy as np
import open3d as o3d
from datetime import datetime
import carla
import grpc
import os
import sys
from google.protobuf import empty_pb2
from google.protobuf import struct_pb2
from optparse import OptionParser
import sys
from concurrent import futures
import time
import random
import threading
sys.path.append('./proto')
sys.path.append('./opencda/scenario_testing/utils/proto')
import carla_pb2_grpc
import carla_pb2

import json

class CarlaAdapter(carla_pb2_grpc.CarlaAdapterServicer):
    def __init__(self, seed, steplength, scenario_params,
                 scenario_manager, cav_list,spectator_transform ,traffic_manager, step_event, stop_event):
        self.scenario_params = scenario_params
        self.scenario_manager = scenario_manager
        self.cav_list = cav_list
        self.spectator_transform = spectator_transform
        self.world = scenario_manager.world
        self.traffic_manager = traffic_manager
        self.client = scenario_manager.client
        self.tick = False
        self.tick_mutex = threading.Lock()
        self.tick_event = threading.Event()  # Aux variable to synchronize the simulation
        self.step_event = step_event  # Aux variable to synchronize the simulation
        self.stop_event = stop_event  # Aux variable to stop the simulation
        print("Steplength: " + str(steplength))

        # Set up the simulator in synchronous mode
        # settings = self.world.get_settings()
        # print(settings)
        # settings.synchronous_mode = True  # Enables synchronous mode
        # settings.fixed_delta_seconds = steplength
        # self.world.apply_settings(settings)

        # Set up the TM in synchronous mode
        ##traffic_manager.set_synchronous_mode(True)

        # Set a seed so behaviour can be repeated if necessary
        ##traffic_manager.set_random_device_seed(seed)
        ##random.seed(seed)
        #if self.scenario_params['scenario']['background_traffic']:
        #    self.generateTraffic(self.scenario_params['scenario']['background_traffic']['vehicle_num'])
        self.steps = 0
        self.time_offset = 0  # Aux variable to keep track of the time offset between CARLA and ms-van3t

    def generateTraffic(self, number):
        spawnPoints = self.world.get_map().get_spawn_points()
        actor_list = self.world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        # discard spawn points that are too close to already spawned vehicles
        for vehicle in vehicles:
            for spawnPoint in spawnPoints:
                if vehicle.get_location().distance(spawnPoint.location) < 5:
                    spawnPoints.remove(spawnPoint)
        for n in range(number):
            randInt = random.randint(0, len(spawnPoints) - 1)
            spawn_point = spawnPoints[randInt]
            spawnPoints.pop(randInt)
            bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
            bp.set_attribute('role_name', 'autopilot')
            vehicle = self.world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, 8000)

    def ExecuteOneTimeStep(self, request, context):
        if self.steps == 0:
            self.time_offset = self.world.get_snapshot().elapsed_seconds
        else:
            ns3_timestamp = self.world.get_snapshot().elapsed_seconds - self.time_offset
            self.world.debug.draw_string(self.spectator_transform.location - carla.Location(x=0,y=3.5),f"NS-3 Simulation Time: {ns3_timestamp:.2f} sec",True,carla.Color(0,0,0,255))
            #+carla.Location(x=30,y=30)
        self.step_event.wait()
        self.step_event.clear()
        if self.stop_event.is_set():
            self.stop_event.clear()
            return carla_pb2.Boolean(value=False)
        #self.world.tick() # OpenCDA should wait for the tick from manual_control
        while not os.path.exists("/tmp/master_ready.flag"):
            time.sleep(0.001)
            continue
        os.remove("/tmp/master_ready.flag")
        print("ONE TIME STEP EXECUTED")
        self.tick_event.set()
        self.steps += 1
        return carla_pb2.Boolean(value=True)

    def GetManagedActorsIds(self, request, context):
        self.world.get_actors()
        actor_list = self.world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        pedestrians = actor_list.filter('walker.*')
        returnValue = carla_pb2.ActorIds()
        for vehicle in vehicles:
            returnValue.actorId.append(vehicle.id)
        for pedestrian in pedestrians:
            returnValue.actorId.append(pedestrian.id)
        return returnValue

    def GetManagedActorById(self, request, context):
        self.world.get_actors()
        actor_list = self.world.get_actors()
        actor = actor_list.find(request.num)
        aLocation = actor.get_location()
        aSpeed = actor.get_velocity()
        aAcceleration = actor.get_acceleration()
        returnValue = carla_pb2.Vehicle()
        returnValue.id = request.num
        returnValue.rolename = actor.attributes['role_name']
        returnValue.location.x = aLocation.x
        returnValue.location.y = aLocation.y
        returnValue.location.z = aLocation.z
        returnValue.speed.x = aSpeed.x
        returnValue.speed.y = aSpeed.y
        returnValue.speed.z = aSpeed.z
        returnValue.acceleration.x = aAcceleration.x
        returnValue.acceleration.y = aAcceleration.y
        returnValue.acceleration.z = aAcceleration.z

        geoPos = self.world.get_map().transform_to_geolocation(aLocation)
        returnValue.latitude = geoPos.latitude
        returnValue.longitude = geoPos.longitude
        returnValue.length = actor.bounding_box.extent.x * 2
        returnValue.width = actor.bounding_box.extent.y * 2
        # print("bounding box: " + str(actor.bounding_box.extent.x) + ", " + str(actor.bounding_box.extent.y))
        # I take abs of lane_id https://github.com/carla-simulator/carla/issues/1469
        returnValue.lane = abs(self.world.get_map().get_waypoint(aLocation).lane_id)
        returnValue.heading = actor.get_transform().rotation.yaw
        # This is redundant, but for CARLA compliance
        returnValue.transform.location.x = actor.get_transform().location.x
        returnValue.transform.location.y = actor.get_transform().location.y
        returnValue.transform.location.z = actor.get_transform().location.z
        returnValue.transform.rotation.pitch = actor.get_transform().rotation.pitch
        returnValue.transform.rotation.yaw = actor.get_transform().rotation.yaw
        returnValue.transform.rotation.roll = actor.get_transform().rotation.roll

        return returnValue

    def GetManagedCAVsIds(self, request, context):
        returnValue = carla_pb2.ActorIds()
        for cav in self.cav_list:
            returnValue.actorId.append(cav.vehicle.id)
        return returnValue

    def GetCarlaWaypoint(self, request, context):
        wpt = self.world.get_map().get_waypoint(carla.Location(x=request.x, y=request.y, z=request.z))
        returnValue = carla_pb2.Waypoint()
        returnValue.location.x = wpt.transform.location.x
        returnValue.location.y = wpt.transform.location.y
        returnValue.location.z = wpt.transform.location.z
        returnValue.rotation.pitch = wpt.transform.rotation.pitch
        returnValue.rotation.yaw = wpt.transform.rotation.yaw
        returnValue.rotation.roll = wpt.transform.rotation.roll
        returnValue.road_id = wpt.road_id
        returnValue.section_id = wpt.section_id
        returnValue.is_junction = wpt.is_junction
        returnValue.lane_id = abs(wpt.lane_id)
        returnValue.lane_width = wpt.lane_width
        returnValue.lane_change = 0
        if wpt.lane_change == carla.LaneChange.Right:
            returnValue.lane_change = 1
        if wpt.lane_change == carla.LaneChange.Left:
            returnValue.lane_change = 2
        if wpt.lane_change == carla.LaneChange.Both:
            returnValue.lane_change = 2

        # # get location of lane change
        # if returnValue.lane_change == 1:
        #     next_wpt = wpt.get_right_lane()
        #     returnValue.lane_change_location.x = next_wpt.transform.location.x
        #     returnValue.lane_change_location.y = next_wpt.transform.location.y
        #     returnValue.lane_change_location.z = next_wpt.transform.location.z
        # elif returnValue.lane_change == 2:
        #     next_wpt = wpt.get_left_lane()
        #     returnValue.lane_change_location.x = next_wpt.transform.location.x
        #     returnValue.lane_change_location.y = next_wpt.transform.location.y
        #     returnValue.lane_change_location.z = next_wpt.transform.location.z
        # else:
        #     returnValue.lane_change_location.x = 0
        #     returnValue.lane_change_location.y = 0
        #     returnValue.lane_change_location.z = 0
        return returnValue

    def GetNextCarlaWaypoint(self, request, context):
        curr_wpt = self.world.get_map().get_waypoint(carla.Location(x=request.x, y=request.y, z=request.z))
        wpt = curr_wpt.next(2)[0]
        returnValue = carla_pb2.Waypoint()
        returnValue.location.x = wpt.transform.location.x
        returnValue.location.y = wpt.transform.location.y
        returnValue.location.z = wpt.transform.location.z
        returnValue.rotation.pitch = wpt.transform.rotation.pitch
        returnValue.rotation.yaw = wpt.transform.rotation.yaw
        returnValue.rotation.roll = wpt.transform.rotation.roll
        returnValue.road_id = wpt.road_id
        returnValue.section_id = wpt.section_id
        returnValue.is_junction = wpt.is_junction
        returnValue.lane_id = abs(wpt.lane_id)
        returnValue.lane_width = wpt.lane_width
        returnValue.lane_change = 0
        if wpt.lane_change == carla.LaneChange.Right:
            returnValue.lane_change = 1
        if wpt.lane_change == carla.LaneChange.Left:
            returnValue.lane_change = 2
        if wpt.lane_change == carla.LaneChange.Both:
            returnValue.lane_change = 2
        return returnValue

    def SetControl(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.id:
                if request.waypoint.x == 0 and request.waypoint.y == 0 and request.waypoint.z == 0:
                    wpt = self.world.get_map().get_waypoint(cav.localizer.get_ego_pos().location)
                    next = wpt.next(max(2, int(cav.localizer.get_ego_spd() / 3.6 * 1)))
                    if len(next) == 0:
                        print("[SetControl] No next waypoint found")
                        return struct_pb2.Value()
                    next_wpt = next[0]
                    # print("Vehicle " + str(cav.vehicle.id) + " next waypoint: " + str(next_wpt.transform.location) +
                    #        "; road_id: " + str(next_wpt.road_id) + "; lane_id: " + str(next_wpt.lane_id)
                    #        + "; is_junction: " + str(next_wpt.is_junction) + "; lane_change: " + str(next_wpt.lane_change))
                    # if wpt.lane_id > next_wpt.lane_id and wpt.is_junction:
                    #     next_wpt = next_wpt.get_left_lane()
                    #     print("Vehicle change to left lane")
                    control = None
                    if request.speed == 1:
                        control = cav.controller.run_step(request.speed * 3.6, next_wpt.transform.location,
                                                          request.acceleration)
                        cav.vehicle.apply_control(control)
                        # print("Vehicle " + str(cav.vehicle.id) + " desired acceleration: " + str(
                        #     request.acceleration) + " control: " + str(control.steer) + ", " + str(
                        #     control.throttle) + ", " + str(control.brake))
                    else:
                        control = cav.controller.run_step(request.speed * 3.6, next_wpt.transform.location, None)
                        cav.vehicle.apply_control(control)
                        # print("Vehicle " + str(cav.vehicle.id) + " desired speed: " + str(
                        #     request.speed) + " control: " + str(control.steer) + ", " + str(
                        #     control.throttle) + ", " + str(control.brake))
                else:
                    transform = carla.Transform(carla.Location(x=request.waypoint.x,
                                                               y=request.waypoint.y,
                                                               z=request.waypoint.z))
                    wpt = self.world.get_map().get_waypoint(transform.location)
                    next_wpt = wpt.next(max(2, int(cav.localizer.get_ego_spd() / 3.6 * 1)))[0]
                    control = cav.controller.run_step(request.speed, next_wpt.transform.location, request.acceleration)

                    cav.vehicle.apply_control(control)
                return struct_pb2.Value()
        return struct_pb2.Value()

    def InsertVehicle(self, request, context):
        blueprint_library = [bp for bp in self.world.get_blueprint_library().filter('vehicle.*')]
        vehicle_bp = blueprint_library[0]
        vehicle_transform = self.world.get_map().get_spawn_points()[2]
        vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
        loc = vehicle.get_location()
        loc.x = request.location.x
        loc.y = request.location.y
        loc.z = request.location.z + 5
        vehicle.set_location(loc)
        number = carla_pb2.Number(num=vehicle.id)
        vehicle.set_autopilot(True, 8000)
        return number

    def GetRandomSpawnPoint(self, request, context):
        randInt = random.randint(0, len(self.world.get_map().get_spawn_points()) - 1)
        spawnPoint = self.world.get_map().get_spawn_points()[randInt]
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(spawnPoint.location +
                                                carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))
        retvalue = carla_pb2.Transform()
        retvalue.location.x = spawnPoint.location.x
        retvalue.location.y = spawnPoint.location.y
        retvalue.location.z = spawnPoint.location.z
        retvalue.rotation.pitch = spawnPoint.rotation.pitch
        retvalue.rotation.yaw = spawnPoint.rotation.yaw
        retvalue.rotation.roll = spawnPoint.rotation.roll
        return retvalue

    def GetActorLDM(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.num:
                if cav.LDM is not None:
                    ego_pos, ego_spd, objects = cav.getInfo()
                    returnValue = carla_pb2.Objects()
                    #if cav.clf_metrics:
                    print(f'gotten metrics: {cav.LDM.get_clf_metrics()}')
                    returnValue.metrics.update(cav.LDM.get_clf_metrics())
                    #returnValue.metrics = cav.LDM.get_clf_metrics()
                    for PO in cav.LDM.getAllPOs():
                        print(f'PO ID: {PO.id}')
                        if (not PO.connected and PO.tracked and PO.onSight) or PO.CPM:
                            print('CPM OR TRACKED WITH ID: ', PO.id)
                            object = carla_pb2.Object()
                            object.id = PO.id
                            if PO.CPM and len(PO.perceivedBy) == 0:
                                #print("CONTINUE FOR ID:",PO.id)
                                continue  # TODO understand why this happens --> it happens because CPM fusion does not set this value
                            # maybe TODO here: CUSTOM ROS MESSAGE CREATION
                            object.dx = PO.perception.xPosition - ego_pos.location.x
                            object.dy = PO.perception.yPosition - ego_pos.location.y
                            object.speed.x = PO.perception.xSpeed
                            object.speed.y = PO.perception.ySpeed
                            object.speed.z = 0
                            object.acceleration.x = PO.perception.xacc
                            object.acceleration.y = PO.perception.yacc
                            object.acceleration.z = 0
                            object.width = PO.perception.width
                            object.length = PO.perception.length
                            object.onSight = PO.onSight
                            object.tracked = PO.tracked
                            object.timestamp = int(1000 * (PO.getLatestPoint().timestamp - self.time_offset))
                            object.label = PO.perception.label + 1  #increasing label by 1 to match ns-3 side labels (in protobuf's params 0 value means no value taken yet)
                            print(f'2. obj with id {object.id} has label {object.label}')
                            object.confidence = PO.perception.confidence
                            if PO.perception.yaw < 0:
                                object.yaw = PO.perception.yaw + 360
                            else:
                                object.yaw = PO.perception.yaw

                            curr_wpt = self.world.get_map().get_waypoint(carla.Location(x=PO.perception.xPosition, y=PO.perception.yPosition, z=0.0))
                            # TODO: this is a workaround, the yaw should be the one from the perception, but pose estimation is not reliable
                            object.yaw = curr_wpt.transform.rotation.yaw
                            object.transform.location.x = PO.perception.xPosition
                            object.transform.location.y = PO.perception.yPosition
                            object.transform.location.z = 0
                            object.transform.rotation.pitch = 0
                            object.transform.rotation.yaw = PO.perception.yaw
                            object.transform.rotation.roll = 0
                            object.detected = not PO.connected
                            if len(PO.perceivedBy) > 0:
                                object.perceivedBy = PO.perceivedBy[0]
                            else:
                                object.perceivedBy = -1
                            returnValue.objects.append(object)

                            if cav.role_name == 'ego':

                                location = {
                                    "x": PO.perception.xPosition,
                                    "y": PO.perception.yPosition,
                                    "z": 0  # Assuming z is always 0 in this case
                                }

                                json_message = {
                                    "dx": PO.perception.xPosition - ego_pos.location.x,
                                    "dy": PO.perception.yPosition - ego_pos.location.y,
                                    "location": location
                                }

                                json_string = json.dumps(json_message, indent=4)

                                print(json_string)

                    print(returnValue.objects)
                    return returnValue

    # message ObjectMinimal {
    #     int32 id = 1;
    # Transform transform = 2;
    # double length = 3;
    # double width = 4;
    # }
    # TODO also get the pedestrians list!!!
    def GetGTaccuracy(self, request, context):
        IoU = 0.0
        obj_o3d_bbx, obj_line_set = get_abs_o3d_bbx(request.transform.location.x, request.transform.location.y,
                                                    request.width, request.length, request.transform.rotation.yaw)
        vehicle_list = {}
        for actor in self.world.get_actors().filter("*vehicle*"):
            id_x = actor.id
            vehicle_list[id_x] = actor
        #add pedestrians to the gt ids
        pedestrian_list = {}
        for actor in self.world.get_actors().filter("*walker*"):
            id_x = actor.id
            pedestrian_list[id_x] = actor
        # TODO: have a distinction between vehicles and pedestrians (propably needed for more clear code and calculations)
        vehicle_list.update(pedestrian_list)

        if request.id in vehicle_list:
            gt = vehicle_list[request.id]
            gt_bbx, gt_line_set = get_abs_o3d_bbx(gt.get_location().x, gt.get_location().y,
                                                  gt.bounding_box.extent.x * 2,
                                                  gt.bounding_box.extent.y * 2, gt.get_transform().rotation.yaw)

            iou = compute_IoU(gt_bbx, obj_o3d_bbx)
            try:
                iou = compute_IoU_lineSet(gt_line_set, obj_line_set)
            except RuntimeError as e:
                # print("Unable to compute the oriented bounding box:", e)
                pass
            if iou > 0.0:
                IoU = iou
        return carla_pb2.DoubleValue(value=IoU)

    def InsertCV(self, request, context):
        init_time = time.time_ns()
        for cav in self.cav_list:
            if cav.vehicle.id == request.egoId:
                newPO = Perception(request.object.transform.location.x,
                                   request.object.transform.location.y,
                                   request.object.width,
                                   request.object.length,
                                   float(request.object.timestamp / 1000) + self.time_offset,
                                   request.object.confidence / 100,
                                   ID=request.object.id)
                newPO.xSpeed = request.object.speed.x
                newPO.ySpeed = request.object.speed.y
                newPO.xacc = request.object.acceleration.x
                newPO.yacc = request.object.acceleration.y
                newPO.heading = request.object.yaw
                newPO.yaw = request.object.yaw
                newPO.id = request.object.id
                cav.ldm_mutex.acquire()
                cav.LDM.CAMfusion(newPO)
                cav.ldm_mutex.release()
                proc_time_us = (time.time_ns() - init_time) / 1000
                proc_time_us = float(proc_time_us)
                return carla_pb2.DoubleValue(value=proc_time_us)

        return carla_pb2.DoubleValue(value=-1.0)

    def InsertObjects(self, request, context):
        init_time = time.time_ns()
        for cav in self.cav_list:
            if cav.vehicle.id == request.egoId:
                toInsert = []
                for obj in request.cpmObjects:
                    newPO = Perception(obj.transform.location.x,
                                       obj.transform.location.y,
                                       obj.width,
                                       obj.length,
                                       float(obj.timestamp / 1000) + self.time_offset,
                                       obj.confidence / 100,
                                       ID=obj.id)
                    newPO.xSpeed = obj.speed.x
                    newPO.ySpeed = obj.speed.y
                    newPO.xacc = obj.acceleration.x
                    newPO.yacc = obj.acceleration.y
                    newPO.heading = obj.yaw
                    newPO.yaw = obj.yaw
                    newPO.id = obj.id
                    toInsert.append(newPO)
                cav.ldm_mutex.acquire()
                cav.LDM.CPMfusion(toInsert, request.fromId)
                cav.ldm_mutex.release()
                proc_time_us = (time.time_ns() - init_time) / 1000
                return carla_pb2.DoubleValue(value=proc_time_us)

        return carla_pb2.DoubleValue(value=-1.0)

    def InsertObject(self, request, context):

        for cav in self.cav_list:
            if cav.vehicle.id == request.egoId:
                newPO = Perception(request.object.transform.location.x,
                                   request.object.transform.location.y,
                                   request.object.width,
                                   request.object.length,
                                   float(request.object.timestamp / 1000) + self.time_offset,
                                   request.object.confidence / 100,
                                   ID=request.object.id)
                newPO.xSpeed = request.object.speed.x
                newPO.ySpeed = request.object.speed.y
                newPO.xacc = request.object.acceleration.x
                newPO.yacc = request.object.acceleration.y
                newPO.heading = request.object.yaw
                newPO.yaw = request.object.yaw
                newPO.id = request.object.id
                if request.object.label == 0:
                    newPO.label = None
                else:
                    newPO.label = request.object.label - 1  #in OpenCDA side, labels:{0:ped,1:veh}
                print(f'3. obj with id {newPO.id} has label {newPO.label}') #will get label None for connected vehicles only (CAService not modified - not essential yet)
                
                toInsert = [newPO]
                cav.ldm_mutex.acquire()
                if request.object.detected:
                    cav.LDM.CPMfusion(toInsert, request.fromId)

                else:
                    cav.LDM.CAMfusion(newPO)
                cav.ldm_mutex.release()
                return carla_pb2.Number(num=1)

        return carla_pb2.Number(num=0)

    def GetCartesian(self, request, context):
        # Compute the CARLA transform with the longitude and latitude values
        ref = self.world.get_map().transform_to_geolocation(
            carla.Location(x=0, y=0, z=0))
        carlaX, carlaY, carlaZ = geo_to_transform(request.x,
                                                  request.y,
                                                  0.0,
                                                  ref.latitude,
                                                  ref.longitude, 0.0)
        returnValue = carla_pb2.Vector()
        returnValue.x = carlaX
        returnValue.y = carlaY
        returnValue.z = carlaZ
        return returnValue

    def GetGeo(self, request, context):
        geoPos = self.world.get_map().transform_to_geolocation(carla.Location(x=request.x,
                                                                              y=request.y,
                                                                              z=request.z))
        return carla_pb2.Vector(x=geoPos.latitude, y=geoPos.longitude, z=geoPos.altitude)

    def hasLDM(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.num:
                if cav.LDM is not None:
                    return carla_pb2.Boolean(value=True)
        return carla_pb2.Boolean(value=False)

    def GetCurrentTime(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return current_time


class MsVan3tCoScenarioManager():
    """
    The manager that controls simulation construction, backgound traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Car;a HD Map.

    """

    def __init__(self, scenario_params, scenario_manager, cav_list, spectator_transform ,traffic_manager, step_event,
                 stop_event,
                 address='localhost', port=1337):
        self.scenario_params = scenario_params
        self.scenario_manager = scenario_manager
        self.traffic_manager = traffic_manager
        self.cav_list = cav_list
        self.spectator_transform = spectator_transform
        self.step_event = step_event
        self.stop_event = stop_event
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
        self.carla_object = CarlaAdapter(1, 0.05, self.scenario_params,
                                         self.scenario_manager,
                                         self.cav_list,
                                         self.spectator_transform,
                                         self.traffic_manager,
                                         self.step_event,
                                         self.stop_event)
        carla_pb2_grpc.add_CarlaAdapterServicer_to_server(self.carla_object, self.server)

        #grpcPort = self.server.add_insecure_port('localhost:1337')
        grpcPort = self.server.add_insecure_port(address + ":" + str(port))
        print("OpenCDA Control Interface running on " + address + ":" + str(grpcPort))

        # write "ready" to a file to signal that the server is ready
        with open("opencdaCI_ready", "w") as f:
            f.write("ready")

        self.server.start()
        print("Server ready")
