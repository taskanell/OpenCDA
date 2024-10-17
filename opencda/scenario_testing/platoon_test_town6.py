import os
from datetime import datetime
import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from opencda.customize.msvan3t.msvan3t_agent import Msvan3tAgent
import time
from opencda.scenario_testing.utils.load_dashboard import LoadDashboard


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        #  Create log directory
        date = datetime.now()
        if opt.pldm:
            dir_name = date.strftime('logs/logsPLDM/log_%d_%m_%H_%M_%S')
        else:
            dir_name = date.strftime('logs/logsLDM/log_%d_%m_%H_%M_%S')
        home_dir = os.path.expanduser('~')
        full_path = os.path.join(home_dir, dir_name)
        os.mkdir(full_path)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create co-simulation scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(['platooning'],
                                                    pldm=opt.pldm,
                                                    log_dir=full_path)

        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='platoon_test_town06',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        spectator_vehicle = single_cav_list[2].vehicle
        last_time = time.time_ns() / 1000000

        transform = spectator_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location +
                                                carla.Location(z=30),
                                                carla.Rotation(pitch=-45)))

        # load_dashboard = LoadDashboard(single_cav_list)

        while True:
            # simulation tick
            scenario_manager.tick()
            # print('Tick: ', time.time_ns() / 1000000 - last_time)
            last_time = time.time_ns() / 1000000
            transform = spectator_vehicle.get_transform()
            transform.location.x += 15
            # spectator.set_transform(carla.Transform(transform.location +
            #                                         carla.Location(z=80),
            #                                         carla.Rotation(pitch=-90)))
            states = []
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                if single_cav.PLDM is not None:
                    states.append(computeCostPLDM(single_cav.PLDM, single_cav.vehicle.id) + single_cav.agent.control_cost)
                elif single_cav.LDM is not None and not single_cav.intruder:
                    states.append(computeCostLDM(single_cav.LDM)+single_cav.agent.control_cost)
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            # load_dashboard.states = states


    finally:
        eval_manager.evaluate()
        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()

def computeCostLDM(LDM):
    POs = LDM.getAllPOs()
    cost = 0
    for PO in POs:
        cost += 1 + len(PO.perceivedBy) * 1
    return cost
def computeCostPLDM(LDM, id):
    POs = LDM.getAllPOs()
    cost = 0
    for PO in POs:
        cost += 1
        if PO.assignedPM == id:
            cost += len(PO.perceivedBy) * 1
    return cost
def run_dashboard():
    LoadDashboard()