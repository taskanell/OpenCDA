# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
from datetime import datetime
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

        #  Create log directory
        date = datetime.now()

        # Convert opt.pmNum to a string
        pm_num_str = str(opt.pmNum)

        if opt.pldm:
            dir_name = date.strftime(f'/home/OpenCDA/logs_single/PLDM/{pm_num_str}_log_%d_%m_%H_%M_%S')
        else:
            dir_name = date.strftime(f'/home/OpenCDA/logs_single/LDM/{pm_num_str}_log_%d_%m_%H_%M_%S')

        os.mkdir(dir_name)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create co-simulation scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   # town='Town06',
                                                   cav_world=cav_world)

        single_cav_list = \
            scenario_manager.create_single_vehicle_manager(application=['platooning'],
                                                           pldm=opt.pldm,
                                                           log_dir=dir_name,
                                                           x=opt.xPos,
                                                           y=opt.yPos)

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
            # time.sleep(0.2)

    finally:
        scenario_manager.close()
        for v in single_cav_list:
            v.destroy()
