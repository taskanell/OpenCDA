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
from opencda.customize.v2x.CAservice import CAservice
from opencda.customize.v2x.CPservice import CPservice
from opencda.customize.v2x.PCservice import PCservice
from opencda.customize.v2x.PLDMservice import PLDMservice

import os, sys, math
import time
import random
import math
import zmq
import json
from threading import Thread
from threading import Event
from proton import Message, Url
from proton.handlers import MessagingHandler
from proton.reactor import Container
# from opencda.customize.core.common.vehicle_manager import LDMObject
from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
import traci  # pylint: disable=import-error
import csv
import time


class Client(MessagingHandler):
    def __init__(self, url, agent):
        super(Client, self).__init__()
        self.url = url
        self.address = "topic://opencda"
        self.agent = agent
        self.run = True
        self.ready = False
        # self.id = agent.vehicle.id

    def on_start(self, event):
        self.conn = event.container.connect(self.url)
        self.sender = event.container.create_sender(self.conn, self.address)
        self.receiver = event.container.create_receiver(self.conn, self.address)

    def cam_sender(self, cam):
        message = Message(body=cam)
        message.properties = {'ETSItype': 'CAM', 'ID': cam['stationID']}

        time.sleep(float(random.randint(0, 50)) / 1000)  # To avoid sync
        self.sender.send(message)

    def cpm_sender(self, cpm):
        message = Message(body=cpm)
        message.properties = {'ETSItype': 'CPM', 'ID': cpm['stationID']}

        time.sleep(float(random.randint(0, 50)) / 1000)  # To avoid sync
        self.sender.send(message)
        # print('Sent ', message.body)

    def platoonControl_sender(self, msg):
        message = Message(body=msg)
        message.properties = {'ETSItype': msg['type'], 'ID': msg['stationID']}

        time.sleep(float(random.randint(0, 50)) / 1000)  # To avoid sync
        self.sender.send(message)

    def on_link_opened(self, event):
        self.ready = True

    def on_message(self, event):
        # kind of BTP implementation
        file_t = time.time_ns() / 1000
        if event.message.properties['ID'] != self.agent.cav.vehicle.id:
            if self.agent.PLDM:
                # print(event.message.properties['ETSItype'])
                if event.message.properties['ETSItype'] == 'PMU' and self.agent.pldmService.leader:
                    new_t, assigned_t = self.agent.pldmService.processPMU(event.message.body)
                    writeLog(self.agent, 'PMU', file_t,
                             len(event.message.body['assignedPOs'])+len(event.message.body['newPOs']),
                             len(self.agent.cav.PLDM.PLDM),
                             PMUnew=new_t,
                             PMUassigned=assigned_t)
                if event.message.properties['ETSItype'] == 'PLU':
                    t_update, t_new, t_genPMU = self.agent.pldmService.processPLU(event.message.body)
                    writeLog(self.agent, 'PLU', file_t, len(event.message.body['perceivedObjects']),
                             len(self.agent.cav.PLDM.PLDM),
                             PLUupdate=t_update,
                             PLUnew=t_new,
                             PLUgenPMU=t_genPMU)
                if event.message.properties['ETSItype'] == 'CPM':
                    t_parse, t_fusion = self.agent.pldmService.processCPM(event.message.body)
                    writeLog(self.agent, 'CPM', file_t, len(event.message.body['perceivedObjects']),
                             len(self.agent.cav.PLDM.PLDM),
                             CPMparse=t_parse,
                             CPMfusion=t_fusion)
                    return
                if event.message.properties['ETSItype'] == 'CAM':
                    self.agent.pldmService.processCAM(event.message.body)
                    writeLog(self.agent, 'CAM', file_t, 1, len(self.agent.cav.PLDM.PLDM))
                    return
            if event.message.properties['ETSItype'] == 'CAM':
                self.agent.caService.processCAM(event.message.body)
                writeLog(self.agent, 'CAM', file_t, 1, len(self.agent.cav.LDM))
            if event.message.properties['ETSItype'] == 'CPM':
                t_parse, t_fusion = self.agent.cpService.processCPM(event.message.body)
                writeLog(self.agent, 'CPM', file_t, len(event.message.body['perceivedObjects']),
                         len(self.agent.cav.LDM),
                         CPMparse=t_parse,
                         CPMfusion=t_fusion)
            if event.message.properties['ETSItype'] == 'PCM':
                self.agent.pcService.processPCM(event.message.body)
            if event.message.properties['ETSItype'] == 'PMM':
                self.agent.pcService.processPMM(event.message.body)


def writeLog(agent, message, timestamp, detected, tracked, PLUupdate=0, PLUnew=0,
             PLUgenPMU=0, PMUassigned=0, PMUnew=0, CPMparse=0, CPMfusion=0):
    with open(agent.file, 'a', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(
            [message,
             (time.time_ns() / 1000) - timestamp,
             detected,
             tracked,
             PLUupdate, PLUnew, PLUgenPMU, PMUassigned, PMUnew, CPMparse, CPMfusion])


class V2XAgent(object):
    """
    V2X agent for CAM,CPM,PMU and PLU creation with Carla information to send messages over AMQP

    """

    # def __init__(self, cav_world, vid, perceptionManager, localizer, carla_map, vehicle):
    def __init__(self, cav, ldm_mutex,
                 AMQPbroker="127.0.0.1:5672", log_dir=None, PLDM=False):

        self.cav = cav
        self.log_dir = log_dir
        self.time = 0

        self.AMQPbroker = AMQPbroker

        self.AMQPhandler = Client(self.AMQPbroker, self)

        self.ldm_mutex = ldm_mutex

        self.amqp_t = Thread(target=self.amqp_thread)
        self.amqp_t.daemon = True
        self.amqp_t.start()

        self.caService = CAservice(cav, self)
        self.cpService = CPservice(cav, self)
        self.pcService = PCservice(cav, self)
        self.PLDM = PLDM
        if self.PLDM:
            self.pldmService = PLDMservice(cav, self)

        if self.PLDM:
            self.file = self.log_dir + '/messages_t_PLDM' + str(self.cav.vehicle.id) + '.csv'
        else:
            self.file = self.log_dir + '/messages_t_LDM' + str(self.cav.vehicle.id) + '.csv'
        with open(self.file, 'w', newline='') as logfile:
            writer = csv.writer(logfile)
            writer.writerow(['Message', 'processingTime', 'detectedPOs', 'trackedPOs', 'PLUupdate', 'PLUnew',
                             'PLUgenPMU', 'PMUassigned', 'PMUnew', 'CPMparse', 'CPMfusion'])

    def amqp_thread(self):
        Container(self.AMQPhandler).run()

    def tick(self):
        self.time = self.cav.time
        # TODO: put both CA and CP services in a separate thread so they don't slow down the simulation
        if self.caService.checkCAMconditions():
            self.AMQPhandler.cam_sender(self.caService.generateCAM())
        if self.cpService.checkCPMconditions():
            self.AMQPhandler.cpm_sender(self.cpService.generateCPM())
        # run platooning step after some time (needed to populate LDM)
        if self.time > 1:
            self.pcService.run_step()
            # print("Vehicle " + str(self.cav.vehicle.id) + ": "
            #      + str(self.pcService.status))
        if self.PLDM:
            self.pldmService.runStep()
