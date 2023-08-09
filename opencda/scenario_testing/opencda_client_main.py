# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from opencda.customize.msvan3t.msvan3t_agent import Msvan3tAgent


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        if scenario_params['sumo']['multiple_clients']:
            scenario_params.scenario.single_cav_list[0].spawn_position[0] -= 10 * (opt.client_number-3)
        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # sumo conifg file path
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')
        # sumo_cfg = os.path.join(current_path,
        #                         '../assets/Town06')
        sumo_cfg = os.path.join(current_path,
                                '../assets/2lane_freeway_simplified')

        # # create co-simulation scenario manager
        # scenario_manager = \
        #     sim_api.CoScenarioManager(scenario_params,
        #                               opt.apply_ml,
        #                               opt.version,
        #                               cav_world=cav_world,
        #                               sumo_file_parent_path=sumo_cfg,
        #                               clients=opt.sumo_clients,
        #                               client_number=opt.client_number)
        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   cav_world=cav_world)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])
        # scenario_manager.setCDAvehicleID(single_cav_list)

        # msvan3tAgent = Msvan3tAgent(cav_world,
        #                             cav_list=single_cav_list,
        #                             cosim_manager=scenario_manager,
        #                             multiple_clients=True)

        spectator = scenario_manager.world.get_spectator()

        while True:
            # simulation tick
            scenario_manager.tick()

            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +
                                                    carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

    finally:
        scenario_manager.close()
        for v in single_cav_list:
            v.destroy()
