# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os
import sys
from omegaconf import OmegaConf
from configparser import ConfigParser

from opencda.version import __version__
home = os.getenv("HOME") + "/git/driving-simulator/"
#sys.path.append(home +"git/driving-simulator/")

def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('-v', "--version", type=str, default='0.9.11',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.11, 0.9.12 is also supported.')
    parser.add_argument('-g', "--gpu", type=int, default=0,
                        help='Specify the GPU id to use, default is 0.')
    parser.add_argument('-s', "--host", type=str, default='localhost',
                        help='Specify the CARLA host ip to connect to, default is localhost.')
    parser.add_argument('-p', "--port", type=int, default=2000,
                        help='Specify the CARLA port to connect to, default is 2000.')

    parser.add_argument('-r', "--grpc_port", type=int, default=1337,
                        help='Specify the CARLA grpc port to connect to, default is 1337.')
    parser.add_argument('-tm', "--tm_port", type=int, default=8000,
                        help='Specify the CARLA traffic manager port to connect to, default is 8000.')
    parser.add_argument('-w', "--pldm" , type=bool, default=False, help='Whether to use the P-LDM')
    parser.add_argument('-i', "--ini_config", required=True, type=str,
                        help='Define the name of the scenario configuration you want to test. This is an ini file'
                        'contained in your ~/git/driving-simulator/data/config folder.')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()
    # print the version of OpenCDA
    #print('Server ready')
    print("OpenCDA Version: %s" % __version__)
    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)
    config_ini = os.path.join(home,'data/config/%s.ini' % opt.ini_config)
    #print("config: ", config_ini)
    ini_parser = ConfigParser()
    ini_parser.read(config_ini)
    ini_dict = {}
    for section in ini_parser.sections():
        ini_dict[section] = dict(ini_parser.items(section))
    #print(ini_dict)
    config_ini = OmegaConf.create(ini_dict)
    #print(config_ini)
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    scene_dict = OmegaConf.merge(default_dict, scene_dict)
    scene_dict = OmegaConf.merge(scene_dict,config_ini)
    OmegaConf.save(scene_dict,"/tmp/scene_dict.yaml")
    #print(scene_dict)
    
    # merge the dictionaries
    #scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # Reorder sys.path: move matching path(s) to the end
    keyword = "carla/dist/carla-0.9.14"
    sys.path[:] = [p for p in sys.path if keyword not in p] + [p for p in sys.path if keyword in p]

    # import the testing script

    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)



    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')
    # run the scenario testing
    scenario_runner(opt, scene_dict)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
