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


class LDMObject:
    def __init__(self, id, xPosition, yPosition, width, length, timestamp, confidence, xSpeed=0, ySpeed=0, heading=0,
                 detected=True, o3d_bbx=None, connected=False):
        self.id = id
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.width = width
        self.length = length
        self.xSpeed = xSpeed
        self.ySpeed = ySpeed
        self.heading = heading
        self.timestamp = timestamp
        self.confidence = confidence
        self.o3d_bbx = o3d_bbx
        self.min = None
        self.max = None
        self.detected = detected
        self.detectedBy = None
        self.connected = connected
        self.estimated_x = 0
        self.estimated_y = 0
        self.estimated_t = 0
        self.CPM_id = None


def sumoToCarlaTransform(x, y, heading, length, width):
    fromTransform = carla.Transform(
        carla.Location(
            x=x, y=y, z=0.0), carla.Rotation(
            pitch=0, yaw=heading, roll=0))
    extent = carla.Vector3D(length / 2.0, width / 2.0, 0.75)
    return BridgeHelper.get_carla_transform(fromTransform,
                                            extent)


class Client(MessagingHandler):
    def __init__(self, url, agent):
        super(Client, self).__init__()
        self.url = url
        self.address = "topic://opencda"
        self.agent = agent
        self.run = True
        self.ready = False

    def on_start(self, event):
        self.conn = event.container.connect(self.url)
        self.sender = event.container.create_sender(self.conn, self.address)
        self.receiver = event.container.create_receiver(self.conn, self.address)

    def req_sender(self):
        # req = Message(reply_to=self.receiver.remote_source.address, body=self.agent.requests.pop())
        req = Message(body=self.agent.requests.pop())
        req.properties = {'type': 'REQUEST'}
        self.sender.send(req)
        print('Sent ', req.body)
        if not self.agent.requests:
            self.agent.request_event.clear()

    def on_link_opened(self, event):
        if event.receiver == self.receiver:
            self.ready = True
            # self.agent.request_event.wait()
            self.req_sender()

    def on_message(self, event):
        if event.message.properties['type'] == 'REPLY':
            print("Received reply %s" % event.message.body)
            if event.message.body['status'] != 'Failed':
                self.agent.replies.append(event.message.body)
                self.agent.reply_event.set()
                self.agent.request_event.wait()
                self.req_sender()
            if not self.agent.run:
                event.connection.close()


class Server(MessagingHandler):
    def __init__(self, url, agent):
        super(Server, self).__init__()
        self.url = url
        self.address = "topic://opencda"
        self.agent = agent

    def on_start(self, event):
        print("Listening on", self.url)
        self.container = event.container
        self.conn = event.container.connect(self.url)
        self.receiver = event.container.create_receiver(self.conn, self.address)
        self.server = self.container.create_sender(self.conn, self.address)

    def on_message(self, event):
        if event.message.properties['type'] == 'REQUEST':
            print("Received", event.message.body)
            reply = self.agent.getCARLAreply(event.message.body)
            print("Status ", reply['status'])
            if reply['status'] != 'Failed':
                amqp_reply = Message(body=reply)
                amqp_reply.properties = {'type': 'REPLY'}
                self.server.send(amqp_reply)


class Msvan3tAgent(object):
    """
    ms-van3t agent for CPM creation with Carla perception.

    Parameters
    ----------
    cav_world : opencda object
        CAV world.

    vid : str
        The corresponding vehicle manager's uuid.

    """

    # def __init__(self, cav_world, vid, perceptionManager, localizer, carla_map, vehicle):
    def __init__(self, cav_world, cosim_manager, cav_list=None, platoon_list=None, openCDAserver=False,
                 AMQPbroker="130.192.238.32:31010", multiple_clients=False, ns3=True):
        self.cosim_manager = weakref.ref(cosim_manager)()
        self.cav_nearby = {}
        if platoon_list is None:
            self.platoon_list = []
        else:
            self.platoon_list = platoon_list
        self.cav_list = cav_list

        # used for cooperative perception.
        self._recieved_buffer = {}

        self.cav_world = weakref.ref(cav_world)()

        self.run = True
        self.server = openCDAserver
        self.AMQPbroker = AMQPbroker

        if multiple_clients:
            if self.server:
                self.requests = []  # List of pending request from ms-van3t to be sent to AMQP broker
                self.request_event = Event()  # Thread event to wait until there are any pending requests
                self.amqpC_t = Thread(target=self.amqpClient_thread)  # AMQP 'client' thread to send ms-van3t's request
                self.amqpC_t.daemon = True
                self.amqpC_t.start()
                # Start ms-van3t thread to forward incoming request to openCDA clients
                self.replies = []
                self.reply_event = Event()
                self.ms_van3t_t = Thread(target=self.ms_van3tM_thread)  # ms-van3t request handler
                self.ms_van3t_t.daemon = True
                self.ms_van3t_t.start()
            else:
                self.amqpC_t = Thread(target=self.amqpServer_thread)  # AMQP 'server' thread to send CARLA's replies
                self.amqpC_t.daemon = True
                self.amqpC_t.start()
        else:
            self.ms_van3t_t = Thread(target=self.ms_van3t_thread)  # ms-van3t request handler
            self.ms_van3t_t.daemon = True
            self.ms_van3t_t.start()

    def amqpClient_thread(self):
        self.request_event.wait()
        Container(Client(self.AMQPbroker, self)).run()

    def amqpServer_thread(self):
        Container(Server(self.AMQPbroker, self)).run()

    def ms_van3tM_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        while self.run:
            #  Wait for next request from client
            # print("Wait for next request from ms-van3t")
            message = socket.recv()

            self.requests.append(json.loads(message))
            self.request_event.set()

            self.reply_event.clear()
            self.reply_event.wait()
            while self.replies:
                reply = self.replies.pop()
                socket.send_json(reply)

    def ms_van3t_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        while self.run:
            #  Wait for next request from client
            # print("Wait for next request from ms-van3t")
            message = socket.recv()

            request = json.loads(message)
            # print("Received request for vehicle " + request['stationID'])
            reply = self.getCARLAreply(request)
            socket.send_json(reply)

    def getCARLAreply(self, request):
        carla2sumo_list = self.cosim_manager.carla2sumo_ids

        carla_id = ""
        for i in carla2sumo_list:
            if carla2sumo_list[i] == request['stationID']:
                carla_id = i

        platoon_member = False

        reply = {'status': 'Failed',
                 'stationID': request['stationID'],
                 'numberOfPOs': 0}

        if carla_id is not "":
            for platoon in self.platoon_list:
                for cav in platoon.vehicle_manager_list:
                    if cav.vehicle.id == carla_id:
                        platoon_member = True
                        #  Send reply back to client
                        if request['status'] == 'request':
                            reply = self.createCPM(cav, request['stationID'])
                            print('Vehicle ' + str(request['stationID']) + ' send CPM')
                        elif request['status'] == 'indication':
                            reply = self.processCPM(cav, request)
                            print('Vehicle ' + str(request['stationID']) + 'received CPM from vehicle carla'
                                  + str(request['from']))
                        elif request['status'] == 'CAM':
                            reply = self.processCAM(cav, request)
                            print('Vehicle ' + str(request['stationID']) + 'received CAM from vehicle carla'
                                  + str(request['from']))

            if not platoon_member:
                for cav in self.cav_list:
                    if cav.vehicle.id == carla_id:
                        # object detection
                        #  Send reply back to client
                        if request['status'] == 'request':
                            reply = self.createCPM(cav, request['stationID'])
                            print('Vehicle ' + str(request['stationID']) + ' send CPM')
                        elif request['status'] == 'indication':
                            reply = self.processCPM(cav, request)
                            print('Vehicle ' + str(request['stationID']) + 'received CPM from vehicle carla'
                                  + str(request['from']))
                        elif request['status'] == 'CAM':
                            reply = self.processCAM(cav, request)
                            print('Vehicle ' + str(request['stationID']) + 'received CAM from vehicle carla'
                                  + str(request['from']))

        return reply

    def processCAM(self, cav, CAM):
        carlaTransform = sumoToCarlaTransform(CAM['fromX'], CAM['fromY'], CAM['fromHeading'], CAM['fromLength'],
                                              CAM['fromWidth'])
        newCV = LDMObject(CAM['from'],
                          carlaTransform.location.x,
                          carlaTransform.location.y,
                          CAM['fromLength'],
                          CAM['fromWidth'],
                          CAM['timestamp'] / 1000,
                          100,
                          CAM['fromSpeed'] * math.cos(math.radians(carlaTransform.rotation.yaw)),
                          CAM['fromSpeed'] * math.sin(math.radians(carlaTransform.rotation.yaw)),
                          carlaTransform.rotation.yaw)
        cav.CAMfusion(newCV)
        return {'status': 'OK',
                'stationID': CAM['stationID']}

    def processCPM(self, cav, CPM):
        newPOs = []
        carlaTransform = sumoToCarlaTransform(CPM['fromX'], CPM['fromY'], CPM['fromHeading'], CPM['fromLength'],
                                              CPM['fromWidth'])
        if 'POs' in CPM:
            for CPMobj in CPM['POs']:
                if CPMobj['ObjectID'] == cav.vehicle.id:
                    continue
                # Convert CPM relative values to absolute CARLA values to then match/fusion with LDMobjects
                dist = math.sqrt(math.pow(CPMobj['xDistance'] / 100, 2) + math.pow(CPMobj['yDistance'] / 100, 2))
                relAngle = math.atan2(CPMobj['yDistance'] / 100, CPMobj['xDistance'] / 100)
                absAngle = relAngle + math.radians(carlaTransform.rotation.yaw)
                xPos = dist * math.cos(absAngle) + carlaTransform.location.x
                yPos = dist * math.sin(absAngle) + carlaTransform.location.y

                dSpeed = math.sqrt(math.pow(CPMobj['xSpeed'] / 100, 2) + math.pow(CPMobj['ySpeed'] / 100, 2))
                relAngle = math.atan2(CPMobj['ySpeed'] / 100, CPMobj['xSpeed'] / 100)
                absAngle = relAngle + math.radians(
                    carlaTransform.rotation.yaw)  # This absolute value is actually the heading of the object
                xSpeed = dSpeed * math.cos(absAngle) + CPM['fromSpeed'] * math.cos(
                    math.radians(carlaTransform.rotation.yaw))
                ySpeed = dSpeed * math.sin(absAngle) + CPM['fromSpeed'] * math.sin(
                    math.radians(carlaTransform.rotation.yaw))

                # CPM object converted to LDM format
                newPO = LDMObject(CPMobj['ObjectID'],
                                  xPos,
                                  yPos,
                                  CPMobj['vehicleLength'] / 10,
                                  CPMobj['vehicleWidth'] / 10,
                                  CPMobj['timestamp'] / 1000,
                                  CPMobj['confidence'])
                newPO.xSpeed = xSpeed
                newPO.ySpeed = ySpeed
                # print('[CPM] ' + str(newPO.id) + ' speed: ' + str(newPO.xSpeed) + ',' + str(newPO.ySpeed))
                newPO.heading = math.degrees(absAngle)
                newPOs.append(newPO)
            cav.CPMfusion(newPOs, CPM['from'])
        return {'status': 'OK',
                'stationID': CPM['stationID']}

    def createCPM(self, cav, sumo_id):
        ego_pos, ego_spd, objects = cav.getInfo()
        LDM = cav.getCPM()

        # For debugging
        # print('Ego position: ' + str(ego_pos.location.x) + ', ' + str(ego_pos.location.y))
        # print('Ego speed: ' + str(ego_spd))

        carla2sumo_list = self.cosim_manager.carla2sumo_ids
        sumo2carla_list = self.cosim_manager.sumo2carla_ids

        # LDM = {}
        POs = []
        nPOs = 0

        for carlaID, LDMobj in LDM.items():
            if LDMobj.detected is False or LDMobj.connected is True:
                continue
            if LDMobj.timestamp < cav.time - 1.0:
                continue
            sumo_POid = ""
            # Get SUMO ID of the CARLA detected objectw
            # for i in sumo2carla_list:
            #     if sumo2carla_list[i] == carlaID:
            #         sumo_POid = i
            # if sumo_POid == "":
            #     if carlaID in carla2sumo_list:
            #         sumo_POid = carla2sumo_list[carlaID]
            #     else:
            #         continue

            dx = (LDMobj.xPosition - ego_pos.location.x)
            dy = (LDMobj.yPosition - ego_pos.location.y)
            dist = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

            relAngle = math.atan2(dy, dx) - math.radians(ego_pos.rotation.yaw)
            xDist = dist * math.cos(relAngle)
            yDist = dist * math.sin(relAngle)

            dxSpeed = (LDMobj.xSpeed - ego_spd / 3.6 * math.cos(math.radians(ego_pos.rotation.yaw)))
            dySpeed = (LDMobj.ySpeed - ego_spd / 3.6 * math.sin(math.radians(ego_pos.rotation.yaw)))
            dSpeed = math.sqrt(math.pow(dxSpeed, 2) + math.pow(dySpeed, 2))
            relAngle = math.atan2(dySpeed, dxSpeed) - math.radians(ego_pos.rotation.yaw)
            xSpeed = dSpeed * math.cos(relAngle)
            ySpeed = dSpeed * math.sin(relAngle)

            # For debugging
            POs.append({'ObjectID': str(LDMobj.id),
                        'Heading': LDMobj.heading * 10,  # In degrees/10
                        'xSpeed': xSpeed * 100,  # Centimeters per second
                        'ySpeed': ySpeed * 100,  # Centimeters per second
                        'vehicleWidth': LDMobj.width * 10,  # In meters/10
                        'vehicleLength': LDMobj.length * 10,  # In meters/10
                        'xDistance': xDist * 100,  # Centimeters
                        'yDistance': yDist * 100,  # Centimeters
                        'confidence': (100 - dist) if dist < 100 else 0,
                        'timestamp': LDMobj.timestamp * 1000000})
            nPOs = nPOs + 1
            if nPOs == 10:
                break

        LDM = {'status': 'OK',
               'stationID': sumo_id,
               'numberOfPOs': nPOs,
               'POs': POs}
        print(LDM)
        return LDM

    def destroy(self):
        self.run = False
        self.ms_van3t_t.join()
