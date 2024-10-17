# -*- coding: utf-8 -*-
import os
import time

import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from threading import Event
from opencda.scenario_testing.utils.ms_van3t_cosim_api import MsVan3tCoScenarioManager


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        cav_world = CavWorld(opt.apply_ml, opt.gpu)

        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   xodr_path=xodr_path,
                                                   cav_world=cav_world,
                                                   carla_host=opt.host,
                                                   carla_port=opt.port)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla(port=opt.tm_port)

        step_event = Event()
        stop_event = Event()
        ms_van3t_manager = \
            MsVan3tCoScenarioManager(scenario_params,
                                     scenario_manager,
                                     single_cav_list,
                                     traffic_manager,
                                     step_event=step_event,
                                     stop_event=stop_event,
                                     port=opt.grpc_port)

        spectator = scenario_manager.world.get_spectator()
        spectator_vehicle = single_cav_list[5].vehicle
        transform = spectator_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location +
                                                carla.Location(z=90),
                                                carla.Rotation(pitch=-90)))
        scenario_manager.tick()

        while True:

            transform = spectator_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +
                                                    carla.Location(z=90),
                                                    carla.Rotation(pitch=-90)))

            init_time = time.time_ns()
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                # control = single_cav.run_step()
                # single_cav.vehicle.apply_control(control)

            print('Time to update LDMs for ', len(single_cav_list), ' vehicles: ', (time.time_ns() - init_time) / 1e6, 'ms')

            step_event.set()
            ms_van3t_manager.carla_object.tick_event.wait()
            ms_van3t_manager.carla_object.tick_event.clear()

    finally:
        scenario_manager.close()
        for v in single_cav_list:
            v.destroy()
