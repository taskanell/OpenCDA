# -*- coding: utf-8 -*-
import os

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

        cav_world = CavWorld(opt.apply_ml)

        if 'name' in scenario_params['scenario']['town']:
            town = scenario_params['scenario']['town']['name']
        else:
            print('No town name has been specified, please check the yaml file.')
            raise ValueError

        # create co-simulation scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town=town,
                                                   cav_world=cav_world,
                                                   carla_host=opt.host,
                                                   carla_port=opt.port)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        step_event = Event()
        stop_event = Event()
        ms_van3t_manager = \
            MsVan3tCoScenarioManager(scenario_params,
                                     scenario_manager,
                                     single_cav_list,
                                     traffic_manager,
                                     step_event=step_event,
                                     stop_event=stop_event)

        spectator = scenario_manager.world.get_spectator()
        spectator_vehicle = single_cav_list[3].vehicle
        transform = spectator_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location +
                                                carla.Location(z=60),
                                                carla.Rotation(pitch=-90)))
        scenario_manager.tick()

        while True:

            transform = spectator_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +
                                                    carla.Location(z=60),
                                                    carla.Rotation(pitch=-90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

            for actor in scenario_manager.world.get_actors().filter("*vehicle*"):
                location = actor.get_location()
                scenario_manager.world.debug.draw_string(location, str(actor.id), False, carla.Color(200, 200, 0))

            step_event.set()
            ms_van3t_manager.carla_object.tick_event.wait()
            ms_van3t_manager.carla_object.tick_event.clear()

    finally:
        stop_event.set() # stop the co-simulation
        step_event.set() # stop the co-simulation
        scenario_manager.close()
        print("Simulation finished.")
        for v in single_cav_list:
            v.destroy()
