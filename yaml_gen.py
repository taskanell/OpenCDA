import yaml
import random

traffic_yaml = """        
# define the background traffic control by carla
carla_traffic_manager:
  global_distance: 4.0 # the minimum distance in meters that vehicles have to keep with the rest
  global_speed_perc: -50
  ignore_lights_percentage: 0 # whether set the traffic ignore traffic lights
  vehicle_list:

"""
base_yaml = '''
vehicle_base:
  sensing:
    perception:
      activate: true # when not activated, objects positions will be retrieved from server directly
      camera:
        visualize: 0 # how many camera images need to be visualized. 0 means no visualization for camera
        num: 4 # how many cameras are mounted on the vehicle. Maximum 3(frontal, left and right cameras)
        # relative positions (x,y,z,yaw) of the camera. len(positions) should be equal to camera num
        positions:
          - [ 2.5, 0, 1.0, 0 ]
          - [ 0.0, 0.3, 1.8, 100 ]
          - [ 0.0, -0.3, 1.8, -100 ]
          - [ -2.0, 0.0, 1.5, 180 ]
      lidar: # lidar sensor configuration, check CARLA sensor reference for more details
        visualize: true

    localization: &base_localize
      activate: false # when not activated, ego position will be retrieved from server directly
      gnss: # gnss sensor configuration
        heading_direction_stddev: 0.1 # degree
        speed_stddev: 0.2
      debug_helper:
        show_animation: false # whether to show real-time trajectory plotting
        x_scale: 10.0 # used to multiply with the x coordinate to make the error on x axis clearer
        y_scale: 10.0 # used to multiply with the y coordinate to make the error on y axis clearer
  map_manager:
    visualize: false
    activate: false
  behavior: &base_behavior
    max_speed: 70 # maximum speed, km/h
    tailgate_speed: 80 # when a vehicles needs to be close to another vehicle asap
    overtake_allowed: false # whether overtake allowed, typically false for platoon leader
    collision_time_ahead: 1.1 # used for collision checking

platoon_base:
  max_capacity: 10
  inter_gap: 0.2 # desired time gap
  open_gap: 1.0 # open gap
  warm_up_speed: 30 # required speed before cooperative merging


scenario:
  single_cav_list: # this is for merging vehicle or single cav without v2x
    - name: cav1
      spawn_position: [121.7194, 139.51, 0.3, 0, 0, 0]
      destination: [606.87, 145.39, 0]
      sensing:
        perception:
          activate: true
          camera:
            visualize: 0 # how many camera images need to be visualized. 0 means no visualization for camera
            num: 4 # how many cameras are mounted on the vehicle. Maximum 3(frontal, left and right cameras)
            # relative positions (x,y,z,yaw) of the camera. len(positions) should be equal to camera num
            positions:
              - [2.5, 0, 1.0, 0]
              - [0.0, 0.3, 1.8, 100]
              - [0.0, -0.3, 1.8, -100]
              - [-2.0, 0.0, 1.5, 180]
          lidar: # lidar sensor configuration, check CARLA sensor reference for more details
            visualize: true
      v2x:
        communication_range: 35
        platoon_init_pos: 1
        pldm: true
      behavior:
        <<: *base_behavior
        max_speed: 70 # maximum speed, km/h
        tailgate_speed: 111
        overtake_allowed: false
        local_planner:
          debug_trajectory: true
          debug: false

    - name: cav2
      spawn_position: [111.7194, 139.51, 0.3, 0, 0, 0]
      destination: [606.87, 145.39, 0]
      sensing:
        perception:
          activate: true
          camera:
            visualize: 0 # how many camera images need to be visualized. 0 means no visualization for camera
            num: 4 # how many cameras are mounted on the vehicle. Maximum 3(frontal, left and right cameras)
            # relative positions (x,y,z,yaw) of the camera. len(positions) should be equal to camera num
            positions:
              - [2.5, 0, 1.0, 0]
              - [0.0, 0.3, 1.8, 100]
              - [0.0, -0.3, 1.8, -100]
              - [-2.0, 0.0, 1.5, 180]
          lidar: # lidar sensor configuration, check CARLA sensor reference for more details
            visualize: true
      v2x:
        communication_range: 35
        platoon_init_pos: 2
        pldm: true
      behavior:
        <<: *base_behavior
        max_speed: 100 # maximum speed, km/h
        tailgate_speed: 111
        overtake_allowed: true
        local_planner:
          debug_trajectory: true
          debug: false

    - name: cav3
      spawn_position: [101.7194, 139.51, 0.3, 0, 0, 0]
      destination: [606.87, 145.39, 0]
      sensing:
        perception:
          activate: true
          camera:
            visualize: 0 # how many camera images need to be visualized. 0 means no visualization for camera
            num: 4 # how many cameras are mounted on the vehicle. Maximum 3(frontal, left and right cameras)
            # relative positions (x,y,z,yaw) of the camera. len(positions) should be equal to camera num
            positions:
              - [2.5, 0, 1.0, 0]
              - [0.0, 0.3, 1.8, 100]
              - [0.0, -0.3, 1.8, -100]
              - [-2.0, 0.0, 1.5, 180]
          lidar: # lidar sensor configuration, check CARLA sensor reference for more details
            visualize: true
      v2x:
        communication_range: 35
        platoon_init_pos: 3
        pldm: true
      behavior:
        <<: *base_behavior
        max_speed: 100 # maximum speed, km/h
        tailgate_speed: 111
        overtake_allowed: true
        local_planner:
          debug_trajectory: true
          debug: false

    - name: cav4
      spawn_position: [91.7194, 139.51, 0.3, 0, 0, 0]
      destination: [606.87, 145.39, 0]
      sensing:
        perception:
          activate: true
          camera:
            visualize: 0 # how many camera images need to be visualized. 0 means no visualization for camera
            num: 4 # how many cameras are mounted on the vehicle. Maximum 3(frontal, left and right cameras)
            # relative positions (x,y,z,yaw) of the camera. len(positions) should be equal to camera num
            positions:
              - [2.5, 0, 1.0, 0]
              - [0.0, 0.3, 1.8, 100]
              - [0.0, -0.3, 1.8, -100]
              - [-2.0, 0.0, 1.5, 180]
          lidar: # lidar sensor configuration, check CARLA sensor reference for more details
            visualize: true
      v2x:
        communication_range: 35
        platoon_init_pos: 4
        pldm: true
      behavior:
        <<: *base_behavior
        max_speed: 100 # maximum speed, km/h
        tailgate_speed: 111
        overtake_allowed: true
        local_planner:
          debug_trajectory: true
          debug: false
'''
y_values = [136.51, 143.51, 146.51, 149.51]

num_files = 10

for i in range(num_files):
    # Load base YAML
    data = yaml.safe_load(traffic_yaml)
    if data["carla_traffic_manager"]["vehicle_list"] is None:
        data["carla_traffic_manager"]["vehicle_list"] = []

    # Generate random number of vehicles
    num_vehicles = random.randint(5, 15)

    # Generate vehicles
    last_x_positions = {y: [] for y in y_values}
    for _ in range(num_vehicles):
        y_value = random.choice(y_values)
        while len(last_x_positions[y_value]) > 6:
            y_value = random.choice(y_values)
        # x_value = max(round(random.uniform(60, 140), 4), last_x_positions[y_value] + 10) # Ensure minimum separation
        x_value = round(random.uniform(50, 160), 4)
        if len(last_x_positions[y_value]) > 0:
            while not all(abs(x_value - previous) >= 10 for previous in last_x_positions[y_value]):
                x_value = round(random.uniform(50, 160), 4)
        # last_x_positions[y_value].append(x_value)
        vehicle = {
            "spawn_position": [
                x_value,  # x value
                y_value,  # y value
                0.3, 0, 0, 0
            ],
            "vehicle_speed_perc": -200
        }
        data["carla_traffic_manager"]["vehicle_list"].append(vehicle)
        last_x_positions[y_value].append(x_value)

    # Write to file
    with open(f'file_{i + 1}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        outfile.write(base_yaml)
