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
                                                   town='Town06',
                                                   cav_world=cav_world)

        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        while True:
            # simulation tick
            time.sleep(0.1)
            scenario_manager.tick()

    finally:
        scenario_manager.close()
