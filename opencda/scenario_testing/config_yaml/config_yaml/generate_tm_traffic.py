import random
import sys

def generate_config(num_vehicles):
    config = """
carla_traffic_manager:
  global_distance: 10.0 # the minimum distance in meters that vehicles have to keep with the rest
  global_speed_perc: -200
  vehicle_list:
"""

    spawn_positions = []

    while len(spawn_positions) < num_vehicles:
        x = random.uniform(300, 600)
        y = random.choice([1.5, 5.5, 9])

        # Check for unique x and y combination and minimum distance
        if not any(abs(x - existing_x) < 10 and y == existing_y for existing_x, existing_y, *_ in spawn_positions):
            spawn_positions.append([x, y, 0.3, 0, 0, 0])

    for pos in spawn_positions:
        # Format each position as a list
        formatted_position = f"    - spawn_position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}, {pos[3]}, {pos[4]}, {pos[5]}]\n"
        config += formatted_position

    return config

def write_config_file(num_vehicles):
    config = generate_config(num_vehicles)
    with open('ms_van3t_platooning_4lane_intruder.yaml', 'w') as new_file:
        new_file.write(config)
        with open('ms_van3t_platooning_4lane_base.yaml', 'r') as base_file:
            new_file.write(base_file.read())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <number_of_vehicles>")
        sys.exit(1)

    try:
        num_vehicles = int(sys.argv[1])
        write_config_file(num_vehicles)
        print(f"Configuration has been written to ms_van3t_platooning_4lane_intruder.yaml")
    except ValueError:
        print("Please enter a valid integer for the number of vehicles.")
        sys.exit(1)
