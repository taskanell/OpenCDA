import docker
import subprocess
from threading import Thread
from threading import Event
import os
import datetime
import time

def calculate_cpu_percent(d):
    cpu_count = len(d["cpu_stats"]["cpu_usage"]["percpu_usage"])
    cpu_percent = 0.0
    cpu_delta = float(d["cpu_stats"]["cpu_usage"]["total_usage"]) - \
                float(d["precpu_stats"]["cpu_usage"]["total_usage"])
    system_delta = float(d["cpu_stats"]["system_cpu_usage"]) - \
                   float(d["precpu_stats"]["system_cpu_usage"])
    if system_delta > 0.0:
        cpu_percent = cpu_delta / system_delta * 100.0 * cpu_count
    return cpu_percent


def get_container_resources(name, event, log_dir):
    # Create a client connection to the Docker daemon
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')

    # List all running containers
    containers = client.containers.list()
    log_file = None
    if log_dir:
        log_file = os.path.join(log_dir, "docker_log.csv")
        with open(log_file, "w") as f:
            f.write("time,cpu_usage,cpu_percent,mem_usage,mem_limit\n")

    while containers and not event.is_set():
        for container in containers:
            # Fetch container stats without streaming to get only the current stats snapshot
            stats = container.stats(stream=False)
            if stats['name'] != name:
                continue
            # Extract resource consumption details
            container_name = stats['name']
            cpu_usage = stats['cpu_stats']['cpu_usage']['total_usage']
            mem_usage = stats['memory_stats']['usage']
            mem_limit = stats['memory_stats']['limit']
            cpu_percent = calculate_cpu_percent(stats)

            print(f"Container: {container_name}")
            print(f"CPU Usage: {cpu_usage}")
            print(f"CPU Percent: {cpu_percent}")
            print(f"Memory Usage: {mem_usage}/{mem_limit}")
            print("------------------------------------------------")

            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"{time.time_ns() / 1000000},{cpu_usage},{cpu_percent},{mem_usage},{mem_limit}\n")
    if event.is_set():
        print("Exiting...")
        return


def run_container(settings):
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')

    # Run the container
    container = client.containers.run(
        image="opencda:latest",
        privileged=True,
        runtime="nvidia",
        name=settings["name"],
        network_mode="host",
        user="opencda",
        volumes=settings["volumes"],
        environment=settings["environment"],
        device_requests=[settings["device_request"]],
        detach=True
    )

    return container

if __name__ == "__main__":

    base_container_settings = {
        "privileged": True,
        "runtime": "nvidia",
        "network_mode": "host",
        "user": "opencda",
        "name": "opencda",
        "volumes": {
            "/mnt/EVO/logs-docker/": {
                "bind": "/home/OpenCDA/logs/",
                "mode": "rw"
            },
            "/tmp/.X11-unix": {
                "bind": "/tmp/.X11-unix",
                "mode": "ro"
        }
        },
        "environment": {
            "DISPLAY": ":1",
            "SCRIPT": "platoon_test_docker",
            "PLDM": "--pldm",
            "ITER": "1",
            "MODE": "pldm"
        },
        "device_request": {
            "Driver": "nvidia",
            "Count": -1,
            "Capabilities": [["gpu"]]
        }
    }

    container = None

    # CARLA settings
    carla_dir = "/mnt/EVO/CARLA_0.9.12"
    carla_command = "./CarlaUE4.sh -prefernvidia"
    process = None
    # Run CARLA
    try:
        process = subprocess.Popen(carla_command.split(), cwd=carla_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("CARLA started.")
    except subprocess.CalledProcessError as e:
        print(f"CARLA failed with error: {e.returncode}")
        print(e.stderr)

    try:
        # PLDM loop
        for i in range(1):
            base_log_dir = "/mnt/EVO/logs-docker/PLDM"
            folders = [f for f in os.listdir(base_log_dir) if
                       f.startswith('log_') and os.path.isdir(os.path.join(base_log_dir, f))]

            # Sort folders based on timestamp
            folders.sort(key=lambda x: datetime.datetime.strptime(x, 'log_%d_%m_%H_%M_%S'), reverse=True)
            latest_folder = folders[0]  # Get the latest folder
            latest_subfolder_path = os.path.join(base_log_dir, latest_folder)

            base_container_settings["environment"]["ITER"] = str(i)
            container = run_container(base_container_settings)
            logger_event = Event()
            logger_thread = Thread(target=get_container_resources, args=(base_container_settings["name"], logger_event, latest_subfolder_path,))
            logger_thread.daemon = True
            logger_thread.start()
            container.wait()
            container.stop()
            logger_event.set()
            container.remove()
            logger_thread.join()
            print("Container removed.")
        # LDM loop
        for i in range(1):
            base_log_dir = "/mnt/EVO/logs-docker/LDM"
            folders = [f for f in os.listdir(base_log_dir) if
                       f.startswith('log_') and os.path.isdir(os.path.join(base_log_dir, f))]

            # Sort folders based on timestamp
            folders.sort(key=lambda x: datetime.datetime.strptime(x, 'log_%d_%m_%H_%M_%S'), reverse=True)
            latest_folder = folders[0]  # Get the latest folder
            latest_subfolder_path = os.path.join(base_log_dir, latest_folder)
            base_container_settings["environment"]["ITER"] = str(i)
            base_container_settings["environment"]["MODE"] = "ldm"
            container = run_container(base_container_settings)
            logger_event = Event()
            logger_thread = Thread(target=get_container_resources, args=(base_container_settings["name"], logger_event,))
            logger_thread.daemon = True
            logger_thread.start()
            container.wait()
            container.stop()
            logger_event.set()
            container.remove()
            logger_thread.join()
            print("Container removed.")

    except KeyboardInterrupt:
        print("Exiting...")
        container.stop()
        container.remove()
        print("Container removed.")
        if process:
            process.terminate()
            print("CARLA terminated.")