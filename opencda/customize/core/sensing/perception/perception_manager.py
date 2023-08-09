# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
import sys
import time

import carla
import cv2
import numpy as np
import open3d as o3d
from collections import deque
import math

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion

from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager


class LDMObject:
    def __init__(self, id, name, xPosition, yPosition, width, length, xSpeed=0, ySpeed=0, heading=0):
        self.id = id
        self.name = name
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.width = width
        self.length = length
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.heading = heading


class ExtendedPerceptionManager(PerceptionManager):
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
        super(ExtendedPerceptionManager, self).__init__(vehicle,
                                                        config_yaml,
                                                        cav_world,
                                                        data_dump,
                                                        carla_world,
                                                        infra_id)
        self.LDM = {}
        self.steps = 0

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

    def detect(self, ego_pos):
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

        if not self.activate:
            objects = self.deactivate_mode(objects)

        else:
            objects = self.activate_mode(objects)

        self.count += 1

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
        yolo_detection = self.ml_manager.object_detector(rgb_images)
        # rgb_images for drawing
        rgb_draw_images = []

        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, self.lidar.data, np.array(
                    rgb_camera.image))
            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion(
                objects,
                yolo_detection.xyxy[i],
                self.lidar.data,
                projected_lidar,
                self.lidar.sensor)

            # calculate the speed. current we retrieve from the server
            # directly.
            self.speed_retrieve(objects)

        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)
            cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def deactivate_mode(self, objects):
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

        vehicle_list = world.get_actors().filter("*vehicle*")
        # todo: hard coded
        thresh = 50 if not self.data_dump else 120

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
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

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
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

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

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

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

    def speed_retrieve(self, objects):
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
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())

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

                    obstacle_vehicle.set_carla_id(v.id)

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
            dot_ve_wp = ve_dir.x * wp_dir.x + \
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

            # Define the function to update LDM

    def update_LDM(self, object_list):
        for obj in object_list:
            if obj.id in self.LDM:
                # If the list of objects for this ID is already at max capacity (10), remove the oldest object
                if len(self.LDM[obj.id]) >= 10:
                    self.LDM[obj.id].popleft()

                # Compute the estimated speed vector based on all previous objects in the deque
                xSpeed_sum = 0
                ySpeed_sum = 0
                for i in range(len(self.LDM[obj.id]) - 1):
                    prev_obj = self.LDM[obj.id][i]
                    curr_obj = self.LDM[obj.id][i + 1]
                    time_diff = 0.1  # Assuming updates occur with a 100ms period
                    xSpeed_sum += (curr_obj.xPosition - prev_obj.xPosition) / time_diff
                    ySpeed_sum += (curr_obj.yPosition - prev_obj.yPosition) / time_diff
                xSpeed = xSpeed_sum / (len(self.LDM[obj.id]) - 1)
                ySpeed = ySpeed_sum / (len(self.LDM[obj.id]) - 1)

                # Compute the estimated heading angle
                heading = math.degrees(math.atan2(ySpeed, xSpeed))
                if heading < 0:
                    heading += 360
                obj.xSpeed = xSpeed
                obj.ySpeed = ySpeed
                obj.heading = heading

                # Compute the average width and length of all objects in the deque
                width_sum = obj.width
                length_sum = obj.length
                for prev_obj in self.LDM[obj.id]:
                    width_sum += prev_obj.width
                    length_sum += prev_obj.length
                avg_width = width_sum / len(self.LDM[obj.id])
                avg_length = length_sum / len(self.LDM[obj.id])
                obj.width = avg_width
                obj.length = avg_length

                self.LDM[obj.id].append(obj)
            else:
                self.LDM[obj.id] = deque([obj], maxlen=10)
