import subprocess
import time

def get_gpu_usage():
    # Command to get GPU usage
    cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"

    # Execute the command
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise Exception("Failed to execute nvidia-smi")


def get_user_by_pid(pid):
    # Command to get user owning the PID
    cmd = f"ps -o user= -p {pid}"

    # Execute the command
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    else:
        return "Unknown or No User"


def main():
    users = ["alamin", "burrello", "carletti", "daghero", "edadmin", "jahier", "motetti", "ponzio", "tredese",
             "benfenati", "cannizzaro", "carlucci", "depaoli", "fiorin", "lezzoche", "mascolini", "pollo", "risso"]
    total_memory = 24564  # This should be dynamic or configured externally

    with open("gpu_usage.csv", "w") as output_file:
        output_file.write("time,GPU0 memory used,GPU1 memory used,GPU2 memory used,"
                          "GPU3 memory used,total memory used [%]," + ",".join(users) + "\n")
        try:
            while True:
                measure = get_gpu_usage()
                usage = [{"pid": m.split(",")[0], "memory": m.split(",")[1].replace(" MiB", "")}
                         for m in measure.split("\n") if m]
                pid_user_map = {u['pid']: get_user_by_pid(u['pid']) for u in usage}

                gpu_memory = [int(u['memory']) for u in usage[:4]]
                new_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{','.join(map(str, gpu_memory))},"
                total = sum(gpu_memory)
                new_line += f"{total / total_memory * 100},"

                for user in users:
                    user_memory = sum(int(u['memory']) for u in usage if pid_user_map.get(u['pid']) == user)
                    new_line += f"{user_memory},"

                output_file.write(new_line[:-1] + "\n")
                output_file.flush()
                print(new_line[:-1])
                time.sleep(5)
        except KeyboardInterrupt:
            print("Monitoring stopped.")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
