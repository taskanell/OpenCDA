# -*- coding: utf-8 -*-

"""Behavior manager for platooning specifically
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
from collections import deque

import carla
import numpy as np
import math
# from opencda.core.application.platooning.fsm import FSM
from opencda.customize.platooning.states import FSM
from opencda.core.application.platooning.platoon_debug_helper import \
    PlatoonDebugHelper
from opencda.core.common.misc import \
    compute_distance, get_speed, cal_distance_angle
from opencda.core.plan.behavior_agent import BehaviorAgent


class PlatooningBehaviorAgentExtended(BehaviorAgent):
    """
    Platoon behavior agent that inherits the single vehicle behavior agent.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla vehicle.

    vehicle_manager : opencda object
        The vehicle manager, used when joining platoon finished.

    v2x_manager : opencda object
        Used to received and deliver information.

    behavior_yaml : dict
        The configuration dictionary for BehaviorAgent.

    platoon_yaml : dict.
        The configuration dictionary for platoon behavior.

    carla_map : carla.Map
        The HD Map used in the simulation.

    Attributes
    ----------
    vehicle_manager : opencda object
        The weak reference of the vehicle manager, used when joining platoon
        finished.

    v2x_manager : opencda object
        The weak reference of the v2x_manager

    debug_helper : opencda Object
        A debug helper used to record the driving performance
         during platooning

    inter_gap : float
        The desired time gap between each platoon member.
    """

    def __init__(
            self,
            vehicle,
            vehicle_manager,
            v2x_manager,
            behavior_yaml,
            platoon_yaml,
            carla_map):

        super(
            PlatooningBehaviorAgentExtended,
            self).__init__(
            vehicle,
            carla_map,
            behavior_yaml)

        self.vehicle_manager = weakref.ref(vehicle_manager)()
        # communication manager
        # self.v2x_manager = weakref.ref(v2x_manager)()
        self.v2xAgent = None

        # used for gap keeping
        self.inter_gap = platoon_yaml['inter_gap']
        # used when open a gap
        self.open_gap = platoon_yaml['open_gap']
        # this is used to control gap opening during cooperative joining
        self.current_gap = self.inter_gap

        # used for merging vehicle
        self.destination_changed = False

        # merging vehicle needs to reach this speed before cooperative merge
        self.warm_up_speed = platoon_yaml['warm_up_speed']

        # used to calculate performance
        self.debug_helper = PlatoonDebugHelper(self.vehicle.id)
        self.time_gap = 100.0
        self.dist_gap = 100.0

        self.control_cost = 0

    def run_step(
            self,
            target_speed=None,
            collision_detector_enabled=True,
            lane_change_allowed=True):
        """
        Run a single step for navigation under platooning agent.
        Finite state machine is used to switch between different
        platooning states.

        Parameters
        ----------
        target_speed : float
            Target speed in km/h

        collision_detector_enabled : bool
            Whether collision detection enabled.

        lane_change_allowed : bool
            Whether lane change is allowed.
        """
        # reset time gap and distance gap record at the beginning
        self.time_gap = 100.0
        self.dist_gap = 100.0

        status = self.v2xAgent.pcService.status
        # case1: the vehicle is not cda enabled
        if status == FSM.DISABLE:
            return super().run_step(target_speed,
                                    collision_detector_enabled)

        # case2: single vehicle keep searching platoon to join
        if status == FSM.SEARCHING or status == FSM.JOIN_REQUEST:
            return super().run_step(target_speed,
                                    collision_detector_enabled)

        # if self.v2xAgent.pcService.front_vehicle is None:
        #     return super().run_step(target_speed,
        #                             collision_detector_enabled)

        # case 4: the merging vehicle selects back joining
        if status == FSM.BACK_JOINING and self.v2xAgent.pcService.front_vehicle:
            self.control_cost = 20
            target_speed, target_waypoint, new_status = \
                self.run_step_back_joining()
            # if joining is finshed
            if new_status == FSM.JOINING_FINISHED:
                self.v2xAgent.pcService.status = FSM.MAINTINING
                # return None, target_speed, target_waypoint
            return target_speed, target_waypoint

        # case 6: leading vehicle behavior
        if status == FSM.LEADING_MODE:
            self.control_cost = 10
            return super().run_step(target_speed, collision_detector_enabled)

        # case7: maintaining status
        if status == FSM.MAINTINING or status == FSM.JOIN_RESPONSE or status == FSM.LEAVE_REQUEST:
            if self.v2xAgent.pcService.front_vehicle:
                return self.run_step_maintaining()
            else:
                target_speed, target_waypoint = super().run_step(target_speed,
                                                                 collision_detector_enabled)
                return None, target_speed, target_waypoint

        # Default
        return super().run_step(target_speed,
                                collision_detector_enabled)

    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update the perception and localization
        information to the behavior agent.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego position from localization module.

        ego_speed : float
            km/h, ego speed.

        objects : dict
            Objects detection results from perception module.
        """
        # update localization information
        self._ego_speed = ego_speed
        self._ego_pos = ego_pos
        self.break_distance = self._ego_speed / 3.6 * self.emergency_param
        # update the localization info to trajectory planner
        self.get_local_planner().update_information(ego_pos, ego_speed)

        # current version only consider about vehicles
        self.objects = objects
        obstacle_vehicles = objects['vehicles']
        self.obstacle_vehicles = self.platoon_list_match(obstacle_vehicles)

        # update the debug helper
        self.debug_helper.update(
            ego_speed,
            self.ttc,
            time_gap=self.time_gap,
            dist_gap=self.dist_gap)

        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            # This method also includes stop signs and intersections.
            self.light_state = str(self.vehicle.get_traffic_light_state())

    def platoon_list_match(self, obstacles):
        """
        Match the detected obstacles with the white list.
        Remove the obstacles that are in white list.
        The white list contains all position of target platoon
        member for joining.

        Parameters
        ----------
        obstacles : list
            A list of carla.Vehicle or ObstacleVehicle

        Returns
        -------
        new_obstacle_list : list
            The new list of obstacles.
        """
        new_obstacle_list = []

        for o in obstacles:
            flag = False
            o_x = o.get_location().x
            o_y = o.get_location().y

            o_waypoint = self._map.get_waypoint(o.get_location())
            o_lane_id = o_waypoint.lane_id

            platoon_map = self.v2xAgent.pcService.PCMap
            for id, pm in platoon_map.items():
                pos = pm[(len(platoon_map[id]) - 1)]['platoonControlContainer']['referencePosition']
                loc = carla.Location(pos["carlaX"], pos["carlaY"], 0)
                vm_x = loc.x
                vm_y = loc.y

                w_waypoint = self._map.get_waypoint(loc)
                w_lane_id = w_waypoint.lane_id

                # if the id is different, then not matched for sure
                if o_lane_id != w_lane_id:
                    continue

                if abs(vm_x - o_x) <= 3.0 and abs(vm_y - o_y) <= 3.0:
                    flag = True
                    break
            if not flag:
                new_obstacle_list.append(o)

        return new_obstacle_list

    def calculate_gap(self, distance):
        """
        Calculate the current vehicle and frontal vehicle's time/distance gap.

        Parameters
        ----------
        distance : float
            Distance between the ego vehicle and frontal vehicle.
        """
        # we need to count the vehicle length in to calculate the gap
        boundingbox = self.vehicle.bounding_box
        veh_length = 2 * abs(boundingbox.location.y - boundingbox.extent.y)

        delta_v = self._ego_speed / 3.6
        time_gap = distance / delta_v
        self.time_gap = time_gap
        self.dist_gap = distance - veh_length

    def platooning_following_manager(self, inter_gap, frontal_vehicle=None):
        """
        Car following behavior in platooning with gap regulation.

        Parameters
        __________
        inter_gap : float
            The gap designed for platooning.
        """

        if frontal_vehicle is None:
            frontal_vehicle = self.v2xAgent.pcService.front_vehicle
        frontal_front_vehicle = self.v2xAgent.pcService.front_front_vehicle

        if len(self._local_planner.get_trajectory()
               ) > self.get_local_planner().trajectory_update_freq - 2:
            return self._local_planner.run_step([], [], [], following=True)
        else:
            # this agent is a behavior agent
            frontal_trajectory = frontal_vehicle['trajectory']

            # get front speed
            frontal_speed = frontal_vehicle['platoonControlContainer']['longitudinalControl'][
                                'longitudinalSpeed'] / 100 * 3.6

            ego_trajetory = deque(maxlen=30)
            ego_loc_x, ego_loc_y, ego_loc_z = \
                self._ego_pos.location.x, \
                    self._ego_pos.location.y, \
                    self._ego_pos.location.z

            # get ego speed
            ego_speed = self._ego_speed

            # compare speed with frontal veh
            frontal_speedd_diff = ego_speed - frontal_speed
            # print('Vehicle: ', self.vehicle.id, 'ego_speed', ego_speed,
            #       '\nfrontal speed: ', frontal_speed,
            #       '\nfrontal speed diff: ', frontal_speedd_diff)

            tracked_length = len(frontal_trajectory) - 1 \
                if not frontal_front_vehicle \
                else len(frontal_trajectory)

            # todo: current not working well on curve
            for i in range(tracked_length):
                delta_t = self.get_local_planner().dt
                # if leader is slowing down(leader target speed is smaller than
                # current speed), use a bigger dt.
                # spd diff max at 15. If diff greater than 8, increase dt
                if frontal_speedd_diff > 3.0:
                    '''
                    # only increase dt when V_ego > V_front (avoid collision)
                    # if V_ego < V_front (diff < 0), stick with small dt
                    # todo: change delta_t to a function:
                    #      --> 1. {V_ego > V_front}: decrease dt to increase
                                  gap, help avoid collision
                    #      --> 2. more difference, more dt adjustment
                    #      --> 3. {V_ego < V_front}: will not collide,
                                  keep default dt to keep gap
                    #      --> 4. {V_ego ~ V_front}: keep default
                                   dt to keep gap
                    '''
                    delta_t = delta_t + frontal_speedd_diff * 0.0125

                if i == 0:
                    pos_x = (frontal_trajectory[i]["x"] +
                             inter_gap / delta_t * ego_loc_x) / (
                                    1 + inter_gap / delta_t)
                    pos_y = (frontal_trajectory[i]["y"] +
                             inter_gap / delta_t * ego_loc_y) / (
                                    1 + inter_gap / delta_t)
                else:
                    pos_x = (frontal_trajectory[i]["x"] +
                             inter_gap / delta_t *
                             ego_trajetory[i - 1][0].location.x) / \
                            (1 + inter_gap / delta_t)
                    pos_y = (frontal_trajectory[i]["y"] +
                             inter_gap / delta_t *
                             ego_trajetory[i - 1][0].location.y) / \
                            (1 + inter_gap / delta_t)

                distance = np.sqrt((pos_x - ego_loc_x) **
                                   2 + (pos_y - ego_loc_y) ** 2)
                velocity = distance / delta_t * 3.6

                ego_trajetory.append([carla.Transform(
                    carla.Location(pos_x,
                                   pos_y,
                                   ego_loc_z)), velocity])

                ego_loc_x = pos_x
                ego_loc_y = pos_y

            if not ego_trajetory:
                wpt = self._map.get_waypoint(self._ego_pos.location)
                next_wpt = wpt.next(max(2, int(self._ego_speed / 3.6 * 1)))[0]
                ego_trajetory.append((next_wpt.transform,
                                      self._ego_speed))

            return self._local_planner.run_step(
                [], [], [], trajectory=ego_trajetory)

    def check_front_obstacle(self, frontal_vehicle):
        min_x = 30
        min_y = 2
        retval = None
        frontal_vehicle_pos = frontal_vehicle['platoonControlContainer']['referencePosition']
        frontal_vehicle_loc = carla.Location(frontal_vehicle_pos["carlaX"], frontal_vehicle_pos["carlaY"], 0)
        front_distance = frontal_vehicle_loc.x - self._ego_pos.location.x
        if self.vehicle_manager.pldm and self.vehicle_manager.PLDM is not None:
            POs = self.vehicle_manager.PLDM.getAllPOs()
        else:
            POs = self.vehicle_manager.LDM.getAllPOs()
        for PO in POs:
            dist_x = PO.perception.xPosition - self._ego_pos.location.x
            dist_y = PO.perception.yPosition - self._ego_pos.location.y
            ego_angle = self.vehicle_manager.localizer.get_ego_pos().rotation.yaw
            angle_rad = ego_angle * math.pi / 180

            # rotated_x = math.cos(angle_rad) * dist_x - math.sin(angle_rad) * dist_y
            # rotated_y = math.sin(angle_rad) * dist_x + math.cos(angle_rad) * dist_y

            if 0 < dist_x <= front_distance and min_y >= dist_y >= -min_y:
                retval = PO
                min_x = dist_x

        if retval:
            referencePosition = {'carlaX': retval.perception.xPosition,
                                 'carlaY': retval.perception.yPosition}
            longControl = {'currentLongitudinalAcceleration': retval.perception.xacc * 100,
                           'longitudinalSpeed': retval.perception.xSpeed * 100}

            platoonControlContainer = {'referencePosition': referencePosition,
                                       'longitudinalControl': longControl,
                                       'vehicleLength': retval.perception.length,
                                       'heading': retval.perception.heading}
            trajectory = []
            # for point in retval.pathHistory:
            #     trajectory.append({'x': point.xPosition, 'y': point.yPosition})
            retval = {'platoonControlContainer': platoonControlContainer,
                      'trajectory': trajectory,
                      'stationID': retval.id}
        return retval

    def run_step_maintaining(self):
        """
        Next step behavior planning for speed maintaining.

        Returns
        -------
        target_speed : float
            The target speed for ego vehicle.

        target_waypoint : carla.waypoint
            The target waypoint for ego vehicle.
        """
        frontal_vehicle = self.v2xAgent.pcService.front_vehicle
        self.current_gap = self.inter_gap

        # Detect potential obstacle
        actual_front = self.check_front_obstacle(frontal_vehicle)
        if actual_front is not None and frontal_vehicle is not None:
            frontal_vehicle = actual_front
        frontal_vehicle_pos = frontal_vehicle['platoonControlContainer']['referencePosition']
        frontal_vehicle_loc = carla.Location(frontal_vehicle_pos["carlaX"], frontal_vehicle_pos["carlaY"], 0)
        ego_vehicle_loc = self._ego_pos.location

        # headway distance
        distance = compute_distance(ego_vehicle_loc, frontal_vehicle_loc)
        # we always use the true position to calculate the timegap for
        # evaluation
        self.calculate_gap(
            compute_distance(ego_vehicle_loc, frontal_vehicle_loc))

        # Distance is computed from the center of the two cars,
        # use bounding boxes to calculate the actual distance
        distance = distance - frontal_vehicle['platoonControlContainer']['vehicleLength'] / 20 - max(
            self.vehicle.bounding_box.extent.y,
            self.vehicle.bounding_box.extent.x)

        # safe control for car following todo: make the coefficient
        # controllable
        if distance <= self._ego_speed / 3.6 * 0.01:
            print(self.vehicle.id, ": emergency stop!")
            return None, 0, None

        target_speed, target_waypoint = self.platooning_following_manager(
            self.inter_gap, frontal_vehicle)

        # Rajamani platoon control
        x_des_acc = None
        if self.v2xAgent.pcService.front_vehicle and self.v2xAgent.pcService.leader_vehicle:
            if self._ego_speed > self.warm_up_speed:
                gap = 10
            else:
                gap = 4
            C1 = 0.7
            xi = 1.0  # Damping ratio
            omega_n = 2.0  # Bandwidth of the controller in Hz

            leader_acc = self.v2xAgent.pcService.leader_vehicle['platoonControlContainer']['longitudinalControl'][
                             'currentLongitudinalAcceleration'] / 100
            leader_speed = self.v2xAgent.pcService.leader_vehicle['platoonControlContainer']['longitudinalControl'][
                               'longitudinalSpeed'] / 100

            front_acc = frontal_vehicle['platoonControlContainer']['longitudinalControl'][
                            'currentLongitudinalAcceleration'] / 100
            front_speed = frontal_vehicle['platoonControlContainer']['longitudinalControl']['longitudinalSpeed'] / 100

            front_length = frontal_vehicle['platoonControlContainer']['vehicleLength'] / 20

            epsilon_i = -(distance) + gap
            ego_speed = self._ego_speed / 3.6
            epsilon_i_dot = ego_speed - front_speed

            # Calculate the control law
            alpha1 = 1 - C1
            alpha2 = C1
            alpha3 = -(2 * xi - C1 * (xi + math.sqrt(xi * xi - 1))) * omega_n
            alpha4 = -C1 * (xi + math.sqrt(xi * xi - 1)) * omega_n
            alpha5 = -(omega_n * omega_n)

            x_des_acc = (alpha1 * front_acc + alpha2 * leader_acc +
                         alpha3 * epsilon_i_dot + alpha4 * (ego_speed - leader_speed) + alpha5 * epsilon_i)
            if actual_front is not None:
                self.control_cost = min(abs(x_des_acc)* 40, 20)
            else:
                self.control_cost = min(abs(x_des_acc) * 5, 20)

            # print('Vehicle: ', self.vehicle.id, ': ego_speed', ego_speed, ', ', 'front_speed: ', front_speed, ', ',
            #       'leader_speed: ', leader_speed, ', ', 'front_acc: ', front_acc, ', ', 'leader_acc: ', leader_acc,
            #       ', ', 'x_des_acc: ', x_des_acc, ', distance: ', distance)
        if self.v2xAgent.pcService.status in (FSM.MAINTINING, FSM.JOIN_RESPONSE, FSM.LEAVE_REQUEST):
            return x_des_acc, target_speed, target_waypoint
        else:
            return target_speed, target_waypoint

    def run_step_back_joining(self):
        """
        Back-joining Algorithm.

        Returns
        -------
        target_speed : float
            The target speed for ego vehicle.

        target_waypoint : carla.waypoint
            The target waypoint for ego vehicle.
        """
        frontal_vehicle = self.v2xAgent.pcService.front_vehicle
        # reset lane change flag every step

        # get necessary information of the ego vehicle and target vehicle in
        # the platooning
        frontal_speed = frontal_vehicle['platoonControlContainer']['longitudinalControl']['longitudinalSpeed'] / 100

        frontal_vehicle_pos = frontal_vehicle['platoonControlContainer']['referencePosition']
        frontal_vehicle_loc = carla.Location(frontal_vehicle_pos["carlaX"], frontal_vehicle_pos["carlaY"], 0)

        frontal_lane = self._map.get_waypoint(frontal_vehicle_loc).lane_id

        # retrieve the platooning's destination
        frontal_destination = carla.Location(630, 141.39, 0.3)

        ego_vehicle_loc = self._ego_pos.location
        ego_wpt = self._map.get_waypoint(ego_vehicle_loc)
        ego_vehicle_lane = ego_wpt.lane_id
        ego_vehicle_yaw = self._ego_pos.rotation.yaw

        distance, angle = \
            cal_distance_angle(
                frontal_vehicle_loc,
                ego_vehicle_loc, ego_vehicle_yaw)

        # calculate the time gap with the frontal vehicle(we use groundtruth
        # position for evaluation)
        self.calculate_gap(compute_distance(frontal_vehicle_loc,
                                            ego_vehicle_loc))

        # 0. make sure the vehicle is behind the ego vehicle
        if angle >= 60 or distance < self._ego_speed / 3.6 * 0.5:
            self.overtake_allowed = False
            print("angle is too large, wait")
            return (
                *
                super().run_step(
                    frontal_speed *
                    0.90,
                    lane_change_allowed=False),
                FSM.BACK_JOINING)

        else:
            self.overtake_allowed = True

        # 1. make sure the speed is warmed up first. Also we don't want to
        # reset destination during lane change
        if self._ego_speed < self.warm_up_speed or \
                self.get_local_planner().potential_curved_road:
            print('warm up speed')
            return (*super().run_step(self.tailgate_speed), FSM.BACK_JOINING)

        if not self.destination_changed:
            print('destination reset!!!!')
            self.destination_changed = True
            self.set_destination(
                ego_wpt.next(4.5)[0].transform.location,
                frontal_destination,
                clean=True,
                clean_history=True)

        # 2. check if there is any other vehicle blocking between ego and
        # platooning

        vehicle_blocking_status = False
        for vehicle in self.obstacle_vehicles:
            vehicle_blocking_status = vehicle_blocking_status or \
                                      self._collision_check.is_in_range(
                                          self._ego_pos,
                                          frontal_vehicle,
                                          vehicle,
                                          self._map,
                                          frontal_vehicle_loc)

        # 3. if no other vehicle is blocking, the ego vehicle is in the
        # same lane with the platooning
        # and it is close enough, then we regard the back joining finished
        if frontal_lane == ego_vehicle_lane \
                and not vehicle_blocking_status \
                and distance < 1.0 * self._ego_speed / 3.6:
            print('joining finished !')
            return (*self.run_step_maintaining(), FSM.JOINING_FINISHED)

        # 4. If vehicle is not blocked, make ego back to the frontal vehicle's
        # lane
        if not vehicle_blocking_status:
            print('no vehicle is blocking!!!')
            if frontal_lane != ego_vehicle_lane:
                left_wpt = ego_wpt.next(
                    max(1.2 * self._ego_speed / 3.6, 5))[0].get_left_lane()
                right_wpt = ego_wpt.next(
                    max(1.2 * self._ego_speed / 3.6, 5))[0].get_right_lane()

                if not left_wpt and not right_wpt:
                    pass
                # if no right lane
                elif not right_wpt:
                    print('take left lane')
                    self.set_destination(
                        left_wpt.transform.location,
                        frontal_destination,
                        clean=True,
                        clean_history=True)
                # if no left lane available
                elif not left_wpt:
                    print('take right lane')
                    self.set_destination(
                        right_wpt.transform.location,
                        frontal_destination,
                        clean=True,
                        clean_history=True)
                # check which lane is closer to the platooning
                elif abs(left_wpt.lane_id - frontal_lane) < \
                        abs(right_wpt.lane_id - frontal_lane):
                    print('take left lane')
                    self.set_destination(
                        left_wpt.transform.location,
                        frontal_destination,
                        clean=True,
                        clean_history=True)
                else:
                    print('take right lane')
                    self.set_destination(
                        right_wpt.transform.location,
                        frontal_destination,
                        clean=True,
                        clean_history=True)

        return (*super().run_step(self.tailgate_speed), FSM.BACK_JOINING)

    def run_step_cut_in_move2point(self):
        """
        The vehicle is trying to get to the move in point.

        Returns
        -------
        target_speed : float
            The target speed for ego vehicle.

        target_waypoint : carla.waypoint
            The target waypoint for ego vehicle.
        """

        frontal_vehicle_manager, rear_vehicle_vm = \
            self.v2x_manager.get_platoon_front_rear()
        frontal_vehicle = frontal_vehicle_manager.vehicle
        frontal_vehicle_speed = \
            frontal_vehicle_manager.v2x_manager.get_ego_speed()

        ego_vehicle_loc = self._ego_pos.location
        ego_vehicle_yaw = self._ego_pos.rotation.yaw

        distance, angle = \
            cal_distance_angle(frontal_vehicle_manager.
                               v2x_manager.get_ego_pos().location,
                               ego_vehicle_loc, ego_vehicle_yaw)

        # calculate the time gap with the frontal vehicle
        self.calculate_gap(compute_distance(frontal_vehicle.get_location(),
                                            ego_vehicle_loc))

        # if there is a obstacle blocking ahead, we just change to back joining
        # mode todo: lane change not considered
        if self.hazard_flag:
            if rear_vehicle_vm:
                rear_vehicle_vm.v2x_manager.set_platoon_status(FSM.MAINTINING)
            # retrieve the last member in the platoon
            platoon_manager, _ = \
                frontal_vehicle_manager.v2x_manager.get_platoon_manager()
            last_member = platoon_manager.vehicle_manager_list[-1]

            # set the last member as the frontal vehicle
            self.v2x_manager.set_platoon_front(last_member)
            self.v2x_manager.set_platoon_rear(None)
            print('switch to back joining!')
            # slow down to join back
            return (*super().run_step(self.max_speed / 2), FSM.BACK_JOINING)

        # the vehicle needs to warm up first. But if the platooning is in car
        # following state, then we should ignore
        if self._ego_speed <= self.warm_up_speed and not \
                frontal_vehicle_manager.agent.car_following_flag:
            print("warming up speed")
            return (*super().run_step(self.tailgate_speed), FSM.MOVE_TO_POINT)

        # if the ego vehicle is still too far away from the front vehicle
        if distance > self._ego_speed / 3.6 * \
                (self.inter_gap + 0.5) and angle <= 80:
            print('trying to get the vehicle')
            return (
                *
                super().run_step(
                    2.0 *
                    frontal_vehicle_speed),
                FSM.MOVE_TO_POINT)

        # if the ego vehicle is too close or exceed the frontal vehicle
        if distance < self._ego_speed / 3.6 * self.inter_gap / 1.5 or \
                angle >= 70:
            print('too close, step back!')
            return (
                *
                super().run_step(
                    0.9 *
                    frontal_vehicle_speed),
                FSM.MOVE_TO_POINT)

        # communicate to the rear vehicle for open gap if rear vehicle exists
        if not rear_vehicle_vm:
            return (
                *self.platooning_merge_management(frontal_vehicle_manager),
                FSM.JOINING)

        distance, angle = cal_distance_angle(
            rear_vehicle_vm.v2x_manager.get_ego_pos().location,
            ego_vehicle_loc, ego_vehicle_yaw)

        # check whether the rear vehicle gives enough gap
        if distance < 1.0 * self.inter_gap / 2.0 * self._ego_speed / 3.6 \
                or angle <= 100 or \
                rear_vehicle_vm.agent.current_gap < self.open_gap:
            # force the rear vehicle open gap for self
            print("too close to rear vehicle!")
            rear_vehicle_vm.v2x_manager.set_platoon_status(FSM.OPEN_GAP)
            return (
                *
                super().run_step(
                    1.5 *
                    frontal_vehicle_speed),
                FSM.MOVE_TO_POINT)

        return (
            *self.platooning_merge_management(frontal_vehicle_manager),
            FSM.JOINING)

    def run_step_cut_in_joining(self):
        """
        Check if the vehicle has been joined successfully.

        Returns
        -------
        target_speed : float
            The target speed for ego vehicle.

        target_waypoint : carla.waypoint
            The target waypoint for ego vehicle.
        """
        print("merging speed %d" % self._ego_speed)

        frontal_vehicle_manager, rear_vehicle_vm = \
            self.v2x_manager.get_platoon_front_rear()

        frontal_vehicle = frontal_vehicle_manager.vehicle
        frontal_vehicle_speed = \
            frontal_vehicle_manager.v2x_manager.get_ego_speed()
        frontal_lane = self._map.get_waypoint(
            frontal_vehicle_manager.v2x_manager.get_ego_pos().location).lane_id

        ego_vehicle_loc = self._ego_pos.location
        ego_vehicle_lane = self._map.get_waypoint(ego_vehicle_loc).lane_id
        ego_vehicle_yaw = self._ego_pos.rotation.yaw

        distance, angle = cal_distance_angle(frontal_vehicle.get_location(),
                                             ego_vehicle_loc, ego_vehicle_yaw)
        # calculate the time gap with the frontal vehicle
        self.calculate_gap(distance)

        if frontal_lane == ego_vehicle_lane and angle <= 5:
            print('merge finished')
            if rear_vehicle_vm:
                rear_vehicle_vm.v2x_manager.set_platoon_status(FSM.MAINTINING)
            return (*self.run_step_maintaining(), FSM.JOINING_FINISHED)

        return (
            *
            super().run_step(
                target_speed=frontal_vehicle_speed,
                collision_detector_enabled=False),
            FSM.JOINING)
