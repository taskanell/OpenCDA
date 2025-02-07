import carla
import numpy as np
import weakref
from opencda.core.application.platooning.platooning_plugin \
    import PlatooningPlugin
from opencda.core.common.misc import compute_distance
from opencda.core.sensing.localization.localization_manager \
    import LocalizationManager
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager

import os, sys, math
import time
import random
import math
import zmq
import json
from threading import Thread
from threading import Event
# from opencda.customize.core.common.vehicle_manager import LDMObject
#from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
#import traci  # pylint: disable=import-error
import csv
import time
import psutil

def writeLog(agent, message, timestamp, detected, tracked, PLUupdate=0, PLUnew=0,
             PLUgenPMU=0, PMUassigned=0, PMUnew=0, CPMparse=0, CPMfusion=0):
    if agent.file:
        with open(agent.file, 'a', newline='') as logfile:
            writer = csv.writer(logfile)
            writer.writerow(
                [message,
                 (time.time_ns() / 1000) - timestamp,
                 detected,
                 tracked,
                 PLUupdate, PLUnew, PLUgenPMU, PMUassigned, PMUnew, CPMparse, CPMfusion])


def sender_zmq(agent):
    while True:
        agent.send_event.wait()
        msg = json.dumps(agent.send_buffer.pop())
        agent.pub_socket.send_string(msg)
        time.sleep(float(random.randint(10, 30)) / 1000)  # To avoid sync messages
        if len(agent.send_buffer) == 0:
            agent.send_event.clear()


def receiver_zmq(agent):
    while True:
        msg = agent.sub_socket.recv_string()
        msg = json.loads(msg)
        # kind of BTP implementation
        file_t = time.time_ns() / 1000
        if msg['stationID'] != agent.cav.vehicle.id:
            if agent.PLDM:
                # print(event.message.properties['ETSItype'])
                if msg['type'] == 'PMU' and agent.pldmService.leader:
                    new_t, assigned_t = agent.pldmService.processPMU(msg)
                    writeLog(agent, 'PMU', file_t,
                             len(msg['assignedPOs']) + len(msg['newPOs']),
                             len(agent.cav.PLDM.PLDM),
                             PMUnew=new_t,
                             PMUassigned=assigned_t)
                if msg['type'] == 'PLU':
                    t_update, t_new, t_genPMU = agent.pldmService.processPLU(msg)
                    writeLog(agent, 'PLU', file_t, len(msg['perceivedObjects']),
                             len(agent.cav.PLDM.PLDM),
                             PLUupdate=t_update,
                             PLUnew=t_new,
                             PLUgenPMU=t_genPMU)
                if msg['type'] == 'CPM':
                    t_parse, t_fusion = agent.pldmService.processCPM(msg)
                    writeLog(agent, 'CPM', file_t, len(msg['perceivedObjects']),
                             len(agent.cav.PLDM.PLDM),
                             CPMparse=t_parse,
                             CPMfusion=t_fusion)
                    continue
                if msg['type'] == 'CAM':
                    agent.pldmService.processCAM(msg)
                    writeLog(agent, 'CAM', file_t, 1, len(agent.cav.PLDM.PLDM))
                    continue
            if msg['type'] == 'CAM':
                agent.caService.processCAM(msg)
                writeLog(agent, 'CAM', file_t, 1, agent.cav.LDM.get_LDM_size())
            if msg['type'] == 'CPM' and agent.intruderApp is None:
                t_parse, t_fusion = agent.cpService.processCPM(msg)
                writeLog(agent, 'CPM', file_t, len(msg['perceivedObjects']),
                         agent.cav.LDM.get_LDM_size(),
                         CPMparse=t_parse,
                         CPMfusion=t_fusion)
            if msg['type'] == 'PCM':
                if agent.pcService is not None:
                    agent.pcService.processPCM(msg)
                elif agent.intruderApp is not None:
                    agent.intruderApp.processPCM(msg)
            if msg['type'] == 'PMM':
                if agent.pcService is not None:
                    agent.pcService.processPMM(msg)


class V2XAgent(object):
    """
    V2X agent for CAM,CPM,PMU and PLU creation with Carla information to send messages over AMQP

    """

    # def __init__(self, cav_world, vid, perceptionManager, localizer, carla_map, vehicle):
    def __init__(self, cav, ldm_mutex, log_dir=None, PLDM=None, rgb_send=True, lidar_send=True):

        self.cav = cav
        self.log_dir = log_dir
        self.time = 0
        self.PLDM = False
        self.file = False

        self.ldm_mutex = ldm_mutex
        self.process = psutil.Process(os.getpid())

        # ZMQ APPROACH
        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.connect(f"tcp://localhost:5555")
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://localhost:5556")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.send_event = Event()
        self.send_buffer = []
        self.sender_thread = Thread(target=sender_zmq, args=(self,))
        self.sender_thread.daemon = True

        self.receiver_thread = Thread(target=receiver_zmq, args=(self,))
        self.receiver_thread.daemon = True
        self.sender_thread.start()
        self.receiver_thread.start()

        self.caService = CAservice(cav, self)
        self.cpService = CPservice(cav, self)
        self.pcService = None
        if self.cav.platooning:
            self.pcService = PCservice(cav, self)

        self.PLDM = PLDM
        if self.PLDM:
            self.pldmService = PLDMservice(cav, self)

        if self.log_dir:
            if self.PLDM:
                self.file = self.log_dir + '/messages_t_PLDM' + str(self.cav.vehicle.id) + '.csv'
                self.cpu_file = self.log_dir + '/cpu_PLDM' + str(self.cav.vehicle.id) + '.csv'
            else:
                self.file = self.log_dir + '/messages_t_LDM' + str(self.cav.vehicle.id) + '.csv'
                self.cpu_file = self.log_dir + '/cpu_LDM' + str(self.cav.vehicle.id) + '.csv'
            with open(self.file, 'w', newline='') as logfile:
                writer = csv.writer(logfile)
                writer.writerow(['Message', 'processingTime', 'detectedPOs', 'trackedPOs', 'PLUupdate', 'PLUnew',
                                 'PLUgenPMU', 'PMUassigned', 'PMUnew', 'CPMparse', 'CPMfusion'])
            with open(self.cpu_file, 'w', newline='') as logfile:
                writer = csv.writer(logfile)
                writer.writerow(['Timestamp', 'messageHandlerCPU', 'messageHandlerRAM', 'overallCPU', 'overallRAM'])
            self.cpu_t = Thread(target=self.cpu_thread)
            self.cpu_t.daemon = True
            self.cpu_t.start()
        else:
            self.file = None

    def cpu_thread(self):
        while True:
            with open(self.cpu_file, 'a', newline='') as logfile:
                writer = csv.writer(logfile)
                writer.writerow([time.time_ns() / 1000,
                                 self.process.cpu_percent(),
                                 self.process.memory_percent(),
                                 psutil.cpu_percent(),
                                 psutil.virtual_memory().percent])
            time.sleep(0.5)


    def tick(self):
        self.time = self.cav.time
        if self.PLDM:
            self.pldmService.runStep()
        # TODO: put both CA and CP services in a separate thread so they don't slow down the simulation
        if self.caService.checkCAMconditions():
            self.send_buffer.append(self.caService.generateCAM())
            self.send_event.set()

        CPM = self.cpService.checkCPMconditions()
        if CPM is not False:
            self.send_buffer.append(self.cpService.generateCPM(CPM))
            self.send_event.set()
        # run platooning step after some time (needed to populate LDM)
        if self.pcService is not None and self.time > 1:
            self.pcService.run_step()
