import psutil
import subprocess
import time
import sys

# Start the process
process = subprocess.Popen(['python3', 'opencda.py', '-t', 'platoon_test_town6', '-v', '0.9.12', '--apply_ml'])

# Get the process info using psutil
p = psutil.Process(process.pid)

try:
    while True:
        # Get the cpu usage
        cpu_usage = p.cpu_percent(interval=1)
        sys.stdout.write(f"\rThe CPU usage of the process is: {cpu_usage}%")
        sys.stdout.flush()
        time.sleep(1)
except psutil.NoSuchProcess:
    print("\nThe process completed or was terminated.")
