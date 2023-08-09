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
import time


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create co-simulation scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   # town='Town06',
                                                   cav_world=cav_world)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['platooning'])

        last_time = time.time_ns() / 1000000



        while True:
            # simulation tick
            print('Tick: ', time.time_ns() / 1000000 - last_time)
            last_time = time.time_ns() / 1000000
            # scenario_manager.tick()
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            time.sleep(0.2)

    finally:
        scenario_manager.close()
        for v in single_cav_list:
            v.destroy()
