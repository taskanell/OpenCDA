# -*- coding: utf-8 -*-
"""
Scenario testing: single vehicle behavior in intersection
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from threading import Event
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from opencda.scenario_testing.utils.ms_van3t_cosim_api import MsVan3tCoScenarioManager



def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        if 'name' in scenario_params['scenario']['town']:
            town = scenario_params['scenario']['town']['name']
        else:
            print('No town name has been specified, please check the yaml file.')
            raise ValueError
            

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town=town,
                                                   cav_world=cav_world,
                                                   carla_host=opt.host,
                                                   carla_port=opt.port)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'],log_dir='/tmp')

        print('I AM HERE')

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla(port=opt.tm_port)
        
        #print(traffic_manager)
        
        step_event = Event()
        stop_event = Event()

        #spectator = scenario_manager.world.get_spectator()
        #transform = single_cav_list[0].vehicle.get_transform()
        transform = carla.Transform(carla.Location(0, 0, 50), carla.Rotation(-90, 0, 0))
        #spectator.set_transform(carla.Transform(
        #    transform.location +
        #        carla.Location(
        #        z=100),
        #        carla.Rotation(
        #            pitch=-
        #           90)))
        
        ms_van3t_manager = \
            MsVan3tCoScenarioManager(scenario_params,
                                     scenario_manager,
                                     single_cav_list,
                                     transform,
                                     traffic_manager,
                                     step_event,
                                     stop_event)

        
        
        #for v in bg_veh_list:
        #bg_veh_list[0].set_autopilot(True,tm_port)            
        scenario_manager.tick()
        # run steps
        while True:
            #scenario_manager.tick()
            #transform = single_cav_list[0].vehicle.get_transform()
            #spectator.set_transform(carla.Transform(
            #    transform.location +
            #    carla.Location(
            #        z=100),
            #    carla.Rotation(
            #        pitch=-
            #        90)))

            for i, single_cav in enumerate(single_cav_list):
                print('RUNNING')
                single_cav.update_info_LDM()
                ego_pos=single_cav.localizer.get_ego_pos().location
                #print("EGO_POS: ", ego_pos)
                #print("ID HERE: ",single_cav.vehicle.id)
                scenario_manager.world.debug.draw_string(ego_pos, str(single_cav.vehicle.id),True,carla.Color(0,0,255,255))
                control = single_cav.run_step()
                if control != None:
                    single_cav.vehicle.apply_control(control)

            #for v in bg_veh_list:
            #     v_pos = v.get_transform().location
            #     scenario_manager.world.debug.draw_string(v_pos, str(v.id), False, carla.Color(200, 200, 0))

            
            step_event.set()
            ms_van3t_manager.carla_object.tick_event.wait()
            ms_van3t_manager.carla_object.tick_event.clear()

    finally:
        #eval_manager.evaluate()

        #if opt.record:
            #scenario_manager.client.stop_recorder()
        stop_event.set()
        step_event.set()
        scenario_manager.close()
        print("Simulation finished.")
        for v in single_cav_list:
            v.destroy()
