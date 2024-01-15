import carla

from opencda.co_simulation.sumo_integration.constants import SPAWN_OFFSET_Z
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from opencda.co_simulation.sumo_integration.constants import INVALID_ACTOR_ID
from opencda.co_simulation.sumo_integration.sumo_simulation import \
    SumoSimulation

import logging
import tempfile
import grpc
import os
import sys
from google.protobuf import empty_pb2
from google.protobuf import struct_pb2
from optparse import OptionParser
import sys

sys.path.append('./proto')
sys.path.append('./opencda/scenario_testing/utils/proto')
import carla_pb2_grpc
import carla_pb2

from concurrent import futures
import time
import random
from datetime import datetime
import threading
from opencda.customize.v2x.aux import Perception
from opencda.core.sensing.localization.coordinate_transform import geo_to_transform
from opencda.scenario_testing.utils.sim_api import ScenarioManager


class CarlaAdapter(carla_pb2_grpc.CarlaAdapterServicer):
    def __init__(self, seed, steplength, scenario_params,
                 scenario_manager, cav_list, traffic_manager, step_event):
        self.scenario_params = scenario_params
        self.scenario_manager = scenario_manager
        self.cav_list = cav_list
        self.world = scenario_manager.world
        self.traffic_manager = traffic_manager
        self.client = scenario_manager.client
        self.tick = False
        self.tick_mutex = threading.Lock()
        self.tick_event = threading.Event()
        self.step_event = step_event
        print("Steplength: " + str(steplength))
        print("Seed: " + str(seed))

        # Set up the simulator in synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = steplength
        self.world.apply_settings(settings)

        # Set up the TM in synchronous mode
        traffic_manager.set_synchronous_mode(True)

        # Set a seed so behaviour can be repeated if necessary
        traffic_manager.set_random_device_seed(seed)
        random.seed(seed)
        if self.scenario_params['scenario']['background_traffic']:
            self.generateTraffic(self.scenario_params['scenario']['background_traffic']['vehicle_num'])
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
        self.step_event.wait()
        self.world.tick()
        self.step_event.clear()
        self.tick_event.set()
        self.steps += 1
        return struct_pb2.Value()

    def GetManagedActorsIds(self, request, context):
        self.world.get_actors()
        actor_list = self.world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        returnValue = carla_pb2.ActorIds()
        for vehicle in vehicles:
            returnValue.actorId.append(vehicle.id)
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

    def SetControl(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.id:
                if request.waypoint.x == 0 and request.waypoint.y == 0 and request.waypoint.z == 0:
                    wpt = self.world.get_map().get_waypoint(cav.localizer.get_ego_pos().location)
                    next_wpt = wpt.next(max(2, int(cav.localizer.get_ego_spd() / 3.6 * 1)))[0]
                    control = None
                    if request.speed == 1:
                        control = cav.controller.run_step(request.speed * 3.6, next_wpt.transform.location, request.acceleration)
                        cav.vehicle.apply_control(control)
                    else:
                        control = cav.controller.run_step(request.speed * 3.6, next_wpt.transform.location, None)
                        cav.vehicle.apply_control(control)
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
        print("return")
        return retvalue

    def GetActorLDM(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.num:
                if cav.LDM is not None:
                    ego_pos, ego_spd, objects = cav.getInfo()
                    returnValue = carla_pb2.Objects()
                    for PO in cav.LDM.getAllPOs():
                        if (PO.detected and PO.tracked and PO.onSight) or PO.CPM:
                            object = carla_pb2.Object()
                            object.id = PO.id
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
                            object.timestamp = int(1000*(PO.getLatestPoint().timestamp-self.time_offset))
                            object.confidence = PO.perception.confidence
                            if PO.perception.yaw < 0:
                                object.yaw = PO.perception.yaw + 360
                            else:
                                object.yaw = PO.perception.yaw
                            object.transform.location.x = PO.perception.xPosition
                            object.transform.location.y = PO.perception.yPosition
                            object.transform.location.z = 0
                            object.transform.rotation.pitch = 0
                            object.transform.rotation.yaw = PO.perception.yaw
                            object.transform.rotation.roll = 0
                            object.detected = PO.detected
                            if PO.CPM and len(PO.perceivedBy) > 0:
                                object.perceivedBy = PO.perceivedBy[0]
                            else:
                                object.perceivedBy = cav.vehicle.id
                            returnValue.objects.append(object)
                    return returnValue

    def InsertObject(self, request, context):
        for cav in self.cav_list:
            if cav.vehicle.id == request.egoId:
                newPO = Perception(request.object.transform.location.x,
                                   request.object.transform.location.y,
                                   request.object.width,
                                   request.object.length,
                                   float(request.object.timestamp/1000) + self.time_offset,
                                   request.object.confidence/100,
                                   ID=request.object.id)
                newPO.xSpeed = request.object.speed.x
                newPO.ySpeed = request.object.speed.y
                newPO.xacc = request.object.acceleration.x
                newPO.yacc = request.object.acceleration.y
                newPO.heading = request.object.yaw
                newPO.yaw = request.object.yaw
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

    def __init__(self, scenario_params, scenario_manager,cav_list,traffic_manager,step_event,
                 address='localhost'):
        self.scenario_params = scenario_params
        self.scenario_manager = scenario_manager
        self.traffic_manager = traffic_manager
        self.cav_list = cav_list
        self.step_event = step_event
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
        self.carla_object = CarlaAdapter(1, 0.05, self.scenario_params,
                                         self.scenario_manager,
                                         self.cav_list,
                                         self.traffic_manager,
                                         self.step_event)
        carla_pb2_grpc.add_CarlaAdapterServicer_to_server(self.carla_object, self.server)

        grpcPort = self.server.add_insecure_port('localhost:1337')
        print("OpenCDA Control Interface running on " + address + ":" + str(grpcPort))
        self.server.start()
