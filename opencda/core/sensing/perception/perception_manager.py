# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
import sys
import time
import math
import carla
import cv2
import numpy as np
import open3d as o3d
import random

import torch

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion
from sklearn.cluster import DBSCAN

class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position,rgb_camera):
        if vehicle is not None:
            world = vehicle.get_world()

        if rgb_camera:
            print("ALREADY SPAWNED CAMERA FROM MC")
            self.sensor = rgb_camera   
        else:

            blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('fov', '100')

            spawn_point = self.spawn_point_estimation(relative_position,
                                                    global_position)
        
            if vehicle is not None:
                self.sensor = world.spawn_actor(
                    blueprint, spawn_point, attach_to=vehicle)
            else:
                self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        self.timstamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        carla_location = carla.Location(x=carla_location.x + x,
                                        y=carla_location.y + y,
                                        z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp


class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position,lidar):
        
        if lidar:
            print("ALREADY SPAWNED LIDAR BY MC")
            self.sensor = lidar 
        else:
            if vehicle is not None:
                world = vehicle.get_world()
            blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

            # set attribute based on the configuration
            blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
            blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
            blueprint.set_attribute('channels', str(config_yaml['channels']))
            blueprint.set_attribute('range', str(config_yaml['range']))
            blueprint.set_attribute(
                'points_per_second', str(
                    config_yaml['points_per_second']))
            blueprint.set_attribute(
                'rotation_frequency', str(
                    config_yaml['rotation_frequency']))
            blueprint.set_attribute(
                'dropoff_general_rate', str(
                    config_yaml['dropoff_general_rate']))
            blueprint.set_attribute(
                'dropoff_intensity_limit', str(
                    config_yaml['dropoff_intensity_limit']))
            blueprint.set_attribute(
                'dropoff_zero_intensity', str(
                    config_yaml['dropoff_zero_intensity']))
            blueprint.set_attribute(
                'noise_stddev', str(
                    config_yaml['noise_stddev']))

            # spawn sensor
            if global_position is None:
                spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
            else:
                spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                            y=global_position[1],
                                                            z=global_position[2]))
            if vehicle is not None:
                self.sensor = world.spawn_actor(
                    blueprint, spawn_point, attach_to=vehicle)
            else:
                self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.data = None
        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        self.points = np.array([data['x'], data['y'], data['z']]).T
        self.obj_tag = np.array(data['ObjTag'])
        self.obj_idx = np.array(data['ObjIdx'])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class RadarSensor:
    """
    Radar manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : list
        Indicates the sensor position relative to vehicle or infrastructure,
        [x, y, z, yaw].

    Attributes
    ----------
    detections : list
        Current list of detected objects.
    sensor : carla.Sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, global_position=None):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.other.radar')
        blueprint.set_attribute('horizontal_fov', '30')
        blueprint.set_attribute('vertical_fov', '30')
        blueprint.set_attribute('range', '100')

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.detections = []
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: RadarSensor._on_radar_event(weak_self, event))

    @staticmethod
    def spawn_point_estimation(relative_position, global_position=None):
        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2]
            )
            pitch = -35

        carla_location = carla.Location(
            x=carla_location.x + x,
            y=carla_location.y + y,
            z=carla_location.z + z
        )

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_radar_event(weak_self, event):
        """Callback method for when radar data is received from the sensor."""
        self = weak_self()
        if not self:
            return

        self.detections = []
        for detection in event:
            detection_data = {
                'velocity': detection.velocity,
                'azimuth': detection.azimuth,
                'altitude': detection.altitude,
                'depth': detection.depth
            }
            self.detections.append(detection_data)


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):
        self.vehicle = vehicle
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self._map = self.carla_world.get_map()
        self.id = infra_id if infra_id is not None else vehicle.id


        self.existing_camera = config_yaml['camera']['existing_camera']
        self.existing_lidar = config_yaml['lidar']['existing_lidar']
        self.activate = config_yaml['activate']
        self.camera_visualize = config_yaml['camera']['visualize']
        self.camera_num = config_yaml['camera']['num']
        self.lidar_visualize = config_yaml['lidar']['visualize']
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None

        self.cav_world = weakref.ref(cav_world)()
        ml_manager = cav_world.ml_manager

        if self.activate and data_dump:
            sys.exit("When you dump data, please deactivate the "
                     "detection function for precise label.")

        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in'
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager
        
        rgb_camera = None
        lidar = None
        print(f"AGENT ROLE NAME: {self.vehicle.attributes['role_name']}")
        if self.existing_camera or self.existing_lidar:
            all_sensors = self.carla_world.get_actors().filter('sensor.*')
            attached_sensors = [sensor for sensor in all_sensors if sensor.parent.id == self.vehicle.id]
            lidar_location = None
            rgb_location = None
            for sensor in attached_sensors:
                if sensor.type_id == 'sensor.camera.rgb':
                    rgb_location = sensor.get_transform().location
                    print(f"RGB with id {sensor.id} has camera location (of VEH {self.vehicle.id}): {rgb_location}" )
                    rgb_camera = sensor
                elif sensor.type_id == 'sensor.lidar.ray_cast':
                    lidar_location = sensor.get_transform().location
                    print(f"Lidar with id {sensor.id} has location (of VEH {self.vehicle.id}): {lidar_location}" )
                    lidar = sensor
                    

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        if self.activate or self.camera_visualize:
            self.rgb_camera = []
            mount_position = config_yaml['camera']['positions']
            if mount_position != None:
                mount_position = config_yaml['camera']['positions']
                assert len(mount_position) == self.camera_num, \
                    "The camera number has to be the same as the length of the" \
                    "relative positions list"
            
            #i want to append first the mc camera
            if rgb_camera:
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, None,
                        self.global_position,rgb_camera))
                #self.camera_visualize += 1
                self.camera_num +=1
                
            #front camera already spawned by mc
            for i in range(self.camera_num if not rgb_camera else self.camera_num-1):
                print('EXTRA HERE')
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position,None))

        else:
            self.rgb_camera = None
        
        print("CAMERA_NUM: ", self.camera_num)

        if lidar:
            self.lidar = LidarSensor(vehicle,
                                    self.carla_world,
                                    config_yaml['lidar'],
                                    self.global_position,
                                    lidar)
        else:
            self.lidar = LidarSensor(vehicle,
                                    self.carla_world,
                                    config_yaml['lidar'],
                                    self.global_position,
                                    None)
        
        #comment local ldm image visualisation init
        '''
        if self.lidar_visualize:
            self.o3d_vis = o3d_visualizer_init(self.id)
        else:
            self.o3d_vis = None
        '''
        self.radar = RadarSensor(vehicle, self.carla_world, self.global_position)

        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if data_dump:
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}
        # traffic light detection related
        self.traffic_thresh = config_yaml['traffic_light_thresh'] \
            if 'traffic_light_thresh' in config_yaml else 50

    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.

        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """
        return a.get_location().distance(self.ego_pos.location)

    def detect(self, ego_pos,role_name):
        """
        Detect surrounding objects. Currently only vehicle detection supported.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos

        objects = {'vehicles': [],
                   'traffic_lights': []}
        
        self.count += 1

        if not self.activate:
            objects = self.deactivate_mode(objects,role_name)
            return objects

        else:
            objects, clf_metrics = self.activate_mode(objects)
            #print(f'metrics: {clf_metrics}')
            return objects, clf_metrics

        #return objects

    def radar_detect(self, objects):

        velocity_range = 7.5  # m/s
        current_rot = self.radar.sensor.get_transform().rotation

        # Step 1: Extract 3D Points
        points = []
        for detect in self.radar.detections:
            azi = math.degrees(detect['azimuth'])
            alt = math.degrees(detect['altitude'])
            fw_vec = carla.Vector3D(x=detect['depth'] - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)
            # point = self.radar.sensor.get_transform().location + fw_vec
            point = fw_vec
            points.append([point.x, point.y, point.z, detect['velocity']])

        if not points:
            return  # No points to process

        # Convert to numpy array
        points_np = np.array(points)

        # Step 2: Cluster the Points
        clustering = DBSCAN(eps=2.0, min_samples=7).fit(points_np)
        labels = clustering.labels_

        # Step 3: Generate Colors for Each Cluster
        unique_labels = set(labels)
        colors = {}
        for label in unique_labels:
            if label == -1:  # Noise points
                colors[label] = (255, 255, 255)  # White
            if label == 0:
                colors[label] = (255, 0, 0)
            if label == 1:
                colors[label] = (0, 255, 0)
            if label == 2:
                colors[label] = (0, 0, 255)

        # Step 4: Draw the Points with Cluster Colors
        for point, label in zip(points, labels):
            #point = point + self.radar.sensor.get_transform().location
            point[0] = point[0] + self.radar.sensor.get_transform().location.x
            point[1] = point[1] + self.radar.sensor.get_transform().location.y
            point[2] = point[2] + self.radar.sensor.get_transform().location.z
            if point[2] < 0.5 or point[2] > 2.5 or point[3] < (5 - self.vehicle.get_velocity().x) or label == -1:
                continue
            # This messes up the yolo detection
            #r, g, b = colors[label]
            # self.carla_world.debug.draw_point(
            #     carla.Location(x=point[0], y=point[1], z=point[2]),
            #     size=0.075,
            #     life_time=0.1,
            #     persistent_lines=False,
            #     color=carla.Color(r, g, b))

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]


        # Step 5: Create and Draw Bounding Boxes
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = points_np[labels == label]
            # skip points with point[0] < 0.5 or point[0] > 2.5 or point[3] < (5 - self.vehicle.get_velocity().x)
            cluster_points = cluster_points[cluster_points[:, 3] > (5 - self.vehicle.get_velocity().x)]

            if len(cluster_points) == 0:
                continue

            min_coords = cluster_points.min(axis=0)
            max_coords = cluster_points.max(axis=0)

            # Create Open3D AxisAlignedBoundingBox
            o3d_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords[:3], max_bound=max_coords[:3])

            # Create the 8 corners of the bounding box
            bbox_corners = [
                [min_coords[0], min_coords[1], min_coords[2]],
                [min_coords[0], min_coords[1], max_coords[2]],
                [min_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], max_coords[1], max_coords[2]],
                [max_coords[0], min_coords[1], min_coords[2]],
                [max_coords[0], min_coords[1], max_coords[2]],
                [max_coords[0], max_coords[1], min_coords[2]],
                [max_coords[0], max_coords[1], max_coords[2]],
            ]

            bbox_corners = np.array(bbox_corners)
            bbox_corners[:, 0] += self.radar.sensor.get_transform().location.x
            bbox_corners[:, 1] += self.radar.sensor.get_transform().location.y
            bbox_corners[:, 2] += self.radar.sensor.get_transform().location.z

            obstacle_vehicle = ObstacleVehicle(bbox_corners, o3d_bbx, confidence=0.71)
            obstacle_vehicle.set_velocity(
                 carla.Vector3D(self.vehicle.get_velocity().x + cluster_points.mean(axis=0)[3], 0, 0))

            for v in vehicle_list:
                loc = v.get_location()
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                    abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.carla_id = v.id

            objects['vehicles'].append(obstacle_vehicle)

        return objects


    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # retrieve current cameras and lidar data
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))


        # yolo detection
        init = time.time_ns()
        ##return_obj = self.ml_manager.object_detector(rgb_images)
        ##print("ITEM: ",return_obj)
        yolo_detection = self.ml_manager.object_detector(rgb_images)
        ##print("RGB IMAGES: ", rgb_images[0].shape)
        ##_,yolo_detection,bboxes_inlane = torch.hub.load(self.ml_manager.src,'yolop', source ='local', mod = self.ml_manager.lane_and_da_detector, mod2=self.ml_manager.object_detector,\
        ##                        image=rgb_images[0], device=self.ml_manager.device,draw_bb_line=False)
        yolo_time = time.time_ns()
        #print('yolo detection time [ms]: ' + str((yolo_time - init) / 1e6))
        #print("IN LANE BBOXES: ",bboxes_inlane)
        # rgb_images for drawing
        rgb_draw_images = []

        data_copy = np.copy(self.lidar.data)

        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, data_copy, np.array(
                    rgb_camera.image))

            rgb_image, projected_radar = st.project_radar_to_camera(
                self.radar.sensor, rgb_camera.sensor, self.radar.detections, np.array(rgb_image))

            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion(
                objects,
                yolo_detection.xyxy[i],
                data_copy,
                projected_lidar,
                self.lidar.sensor,
                self.ego_pos)
            
            print("YOLO OBJS: ",objects)

            # calculate the speed. current we retrieve from the server
            # directly.
            # THIS WORKS ONLY FOR ONE CAMERA!!
            # TODO: check what to showcase as local clasifications (before or after the following confidence and duplicate filtering)
            # TODO: Give confidence level as a parameter from the ini config file
            objects['vehicles'] = [item for item in objects['vehicles'] if item.confidence >= 0.7 or (item.label==0 and item.confidence >= 0.35)] #or (item.label==0 and item.confidence>=0.3)] #if (item.confidence >= 0.7) or 
            metrics_dict = self.speed_retrieve_and_classify_metrics(objects)

        self.radar_detect(objects)


        fusion_time = time.time_ns()
        #print('fusion time [ms]: ' + str((fusion_time - yolo_time) / 1e6))
        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.8, fy=0.8)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)
            cv2.waitKey(1)

        
        #objects['vehicles'] = [item for item in objects['vehicles'] if item.confidence >= 0.7 or (item.label==0 and item.confidence >= 0.35)] #or (item.label==0 and item.confidence>=0.3)] #if (item.confidence >= 0.7) or 
        print(f'SENT OBJS: {objects}')
        duplicate_indices = set()
        # Iterate through the objects to check for duplicates
        for i in range(len(objects['vehicles'])):
            for j in range(i + 1, len(objects['vehicles'])):
                dist = math.sqrt(pow(objects['vehicles'][i].location.x - objects['vehicles'][j].location.x, 2)
                                 + pow(objects['vehicles'][i].location.y - objects['vehicles'][j].location.y, 2))
                print(objects['vehicles'][i].carla_id,objects['vehicles'][j].carla_id)
                if objects['vehicles'][i].carla_id == objects['vehicles'][j].carla_id: #dist < 3 or
                    # if (objects['vehicles'][i].bounding_box.extent.x*objects['vehicles'][i].bounding_box.extent.y) > \
                    #         (objects['vehicles'][j].bounding_box.extent.x * objects['vehicles'][j].bounding_box.extent.y):
                    if objects['vehicles'][i].confidence > objects['vehicles'][j].confidence:
                        duplicate_indices.add(j)
                    else:
                        duplicate_indices.add(i)
        print(duplicate_indices)
        # Remove duplicate objects from the list
        for index in sorted(duplicate_indices, reverse=True):
            objects['vehicles'].pop(index)
        #comment local ldm image visualisation
        '''
        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(data_copy, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        '''
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects
        for obj in objects['vehicles']:
            if obj.carla_id == -1:
                print(f'NOT CARLA ID OBJ WITH LABEL {obj.label}')

        #print('Matching time [ms]: ' + str((time.time_ns() - fusion_time) / 1e6))
        return objects, metrics_dict

    def deactivate_mode(self, objects,role_name):
        """
        Object detection using server information directly.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = list(world.get_actors().filter("*vehicle*"))
        pedestrian_list = list(world.get_actors().filter("*walker*")) 
        # todo: hard coded
        thresh = 30 if not self.data_dump else 30

        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                        v.id != self.id]

        # use semantic lidar to filter out vehicles out of the range
        if self.data_dump:
            vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids,label=1) for v in vehicle_list]
            if role_name != 'ego':
                pedestrian_list = [
                    ObstacleVehicle(
                        None,
                        None,
                        v,
                        self.lidar.sensor,
                        self.cav_world.sumo2carla_ids,label=0) for v in pedestrian_list]
            
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        # add the pedestrian list to the ego's perception only if the role name is not ego
        # useful for the CAVs to detect pedestrians with GT data
        if role_name != 'ego':
            vehicle_list = vehicle_list + pedestrian_list
        objects.update({'vehicles': vehicle_list})

        if self.camera_visualize:
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.7, fy=0.7)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)
        '''
        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        '''

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def getGTobjects(self):
        """
        Object detection using server information directly.

        Returns
        -------
         objects: dict
            Object dictionary.
        """
        world = self.carla_world
        # TODO also get the pedestrians list
        vehicle_list = world.get_actors().filter("*vehicle*")
        # todo: hard coded
        thresh = 75

        if self.ego_pos:
            vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                            v.id != self.id]
        else:
            vehicle_list = [v for v in vehicle_list if v.id != self.id]

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    None) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects = {'vehicles': vehicle_list}
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)

        return objects

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.

        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        # label 10 is the vehicle
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image


    def check_for_previous_mindist_object(self, obstacle_vehicle, appended_ids, TP):
        """
        Check if the object has a previous min distance object assigned.
        If yes, assign the previous object with the right after min distance
        object. If no, decrease the TP as the object is not assigned to any
        vehicle id.

        Parameters
        ----------
        obstacle_vehicle : ObstacleVehicle
            The detected obstacle vehicle.
        appended_ids : dict
            The dictionary that contains the appended vehicle ids given by Carla server.
        TP : int
            The number of true positive vehicles.
        Returns
        -------
        TP : int
            The updated number of true positive vehicles.
        """

        previous_min_obj = appended_ids[obstacle_vehicle.carla_id][2]
        if previous_min_obj != -1:
            print(f'Update id {obstacle_vehicle.carla_id} vehicle with previous min object')
            appended_ids[obstacle_vehicle.carla_id][1] = previous_min_obj  # assign the previous object with the right after min distance
            previous_min_obj.set_carla_id(obstacle_vehicle.carla_id) # set the former v.id to the previous object
        else:
            print(f'No object currently assigned object to {obstacle_vehicle.carla_id} id vehicle')
            print("Decreasing TP..")
            TP-=1
        return TP

    def speed_retrieve_and_classify_metrics(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 100 and
                        v.id != self.id]
        # TODO: have a better check for more than one pedestrian
        ped = world.get_actors().filter("walker.pedestrian*")
        # ped = [v for v in ped if self.dist(v) < 100 and v.id != self.id] # maybe needed in the future, for now i just want to get the ped id
        ped = [v for v in ped]
        actor_list = ped + vehicle_list
        ped_ids = [p.id for p in ped]
        det_ped = 0
        appended_ids = {}
        num_of_peds = len(list(ped))
        num_of_vehs = len(list(vehicle_list))
        total_actors = num_of_peds+num_of_vehs
        print(f'Detected objects: {objects["vehicles"]}')
        num_of_detobj = len(objects['vehicles'])


        if objects.get('static', []):
            num_of_detobj += len(objects['static']) 
        TP = 0
        FN = 0

        # todo: consider the minimum distance to be safer in next version (comment by the ms-van3t team)
        #for v in vehicle_list:
        for v in actor_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                # TODO: Investigate if this tactic is efficient. If the vehicle is stable, it would propably return zero speed.
                # It also adds carla id value with hard coded distance of gt vehicle to detected object (less than 3) --> efficient?
                # if obstacle_speed > 0: 
                #     print(f'already matched {obstacle_vehicle.carla_id}')
                #     continue
                obstacle_loc = obstacle_vehicle.get_location()
                # TODO : For more than one pedestrian i should have a check too to give id based on distance (i avoid it for now because i get None id values for peds sometimes)
                if obstacle_vehicle.label == 0 and v.id in ped_ids:
                    print(f'ped id detected')
                    if det_ped<num_of_peds:
                        det_ped += 1
                        ped_id = ped[0].id
                        TP+=1
                        print(f'[1] TP with id {ped_id} pedestrian')
                    else:
                        if obstacle_vehicle.carla_id == -1 and num_of_peds!=0:
                            print(f'already matched ped id {ped_id}')
                    if num_of_peds!=0:
                        obstacle_vehicle.set_carla_id(ped_id)
                    #continue
                print(f'VALUE X: {abs(loc.x - obstacle_loc.x)}, VALUE Y: {abs(loc.y - obstacle_loc.y)} for {obstacle_vehicle.label}')
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())
                    # temporary solution to avoid comparing ped ids with vehicle objects. TODO: have a better check for more than one pedestrian to assign velocity
                    if v.id in ped_ids:
                        continue
                    dist_sq = (loc.x - obstacle_loc.x) ** 2 + (loc.y - obstacle_loc.y) ** 2

                    # the case where the obstacle vehicle is controled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)
                    #if obstacle_vehicle.carla_id != -1:
                    if obstacle_vehicle.label == 1:

                        if v.id not in appended_ids: 
                            #appended_ids.append(v.id)
                            #obstacle_vehicle.set_carla_id(v.id)
                            if obstacle_vehicle.carla_id != -1:
                                if appended_ids[obstacle_vehicle.carla_id][0] > dist_sq:
                                    print(f'Assigning current id {v.id} vehicle to already assigned object')
                                    TP = self.check_for_previous_mindist_object(obstacle_vehicle, appended_ids, TP)
                                else: 
                                    continue
                            appended_ids[v.id] = (dist_sq,obstacle_vehicle,-1)
                            print(f"[2] TP with {v.id} vehicle")
                            TP+=1

                        # if the distance is smaller than the previous detected, then update the min distance and object assigned to this v.id
                        if dist_sq < appended_ids[v.id][0]:
                            if obstacle_vehicle.carla_id != -1:
                                print(f'Assigning current id {v.id} vehicle to already assigned object')
                                TP = self.check_for_previous_mindist_object(obstacle_vehicle, appended_ids, TP)
                            # keep the previous object with the right after min distance
                            # and update the min distance and object assigned to this v.id
                            appended_ids[v.id] = (dist_sq,obstacle_vehicle,appended_ids[v.id][1]) 

            if v.id in appended_ids:       
                appended_ids[v.id][1].set_carla_id(v.id)

        print(f'APPENDED IDS: {appended_ids.keys()}')
        
        # check it the static (non ped or veh actors) objects are misperceived as vehicles or pedestrians
        if objects.get('static', []):
            print(f"static obj: {len(objects['static'])}")
            for v in vehicle_list:
                if v.id not in appended_ids:
                    loc = v.get_location()
                    for obstacle_vehicle in objects['static']:
                            obstacle_loc = obstacle_vehicle.get_location()
                            if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                            abs(loc.y - obstacle_loc.y) <= 3.0:
                                print(f'Object with label {obstacle_vehicle.label} FP')
                                TP -= 1

        
        print(f'FP:{num_of_detobj-TP}, TP:{TP}, FN:{total_actors-TP}, TotalDetections:{num_of_detobj}, TotalActors:{total_actors}')
        return {'TP': TP, 'FP': num_of_detobj-TP, 'FN': total_actors-TP, 'TotalDetections':num_of_detobj, 'TotalActors':total_actors}
                      

    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        if self.ego_pos:
            vehicle_location = self.ego_pos.location
            vehicle_waypoint = self._map.get_waypoint(vehicle_location)

            activate_tl, light_trigger_location = \
                self._get_active_light(tl_list, vehicle_location, vehicle_waypoint)

            objects.update({'traffic_lights': []})

            if activate_tl is not None:
                traffic_light = TrafficLight(activate_tl,
                                             light_trigger_location,
                                             activate_tl.get_state())
                objects['traffic_lights'].append(traffic_light)

        return objects

    def _get_active_light(self, tl_list, vehicle_location, vehicle_waypoint):
        for tl in tl_list:
            object_location = \
                TrafficLight.get_trafficlight_trigger_location(tl)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x +\
                        ve_dir.y * wp_dir.y + \
                        ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue
            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            return tl, object_waypoint.transform.location

        return None, None

    def destroy(self):
        """
        Destroy sensors.
        """
        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()
