import matplotlib.pyplot as plt
import psutil
from matplotlib.animation import FuncAnimation
import numpy as np
from threading import Thread
from threading import Event
import math

# Function to determine bar color based on CPU load
# def get_color(cpu_load):
#     if cpu_load < 50:
#         return 'green'
#     elif 50 <= cpu_load < 75:
#         return 'yellow'
#     else:
#         return 'red'

def get_color(cpu_load):
    colors = [
        '#00FF00', '#19FF00', '#33FF00', '#4CFF00', '#66FF00',
        '#7FFF00', '#99FF00', '#B2FF00', '#CCFF00', '#E5FF00',
        '#FFFF00', '#FFCC00', '#FF9900', '#FF6600', '#FF3300',
        '#FF3300', '#FF3300', '#FF0000', '#FF0000', '#FF0000'
    ]

    # The list starts with saturated green and transitions through to the most saturated red.

    # Map the load to a range from 0 to 19 (for 20 colors)
    index = int((cpu_load / 100) * (len(colors) - 1))

    # Ensure the index is within the bounds
    index = max(0, min(index, len(colors) - 1))

    return colors[index]


def scale_value(value):
    if value >= 15:
        return 100
    scaled_value = (value / 14) * 100

    return math.trunc(scaled_value)

# This function will be called repeatedly to animate the dashboard
def animate(i, dash):
    plt.cla()  # Clear the current axes.

    # Get the CPU percentage
    #cpu_percents = psutil.cpu_percent(interval=None, percpu=True)
    cpu_percents = [min(state, 100) for state in dash.states]
    # cpu_percents = [scale_value(state) for state in dash.states]
    bar_colors = [get_color(load) for load in cpu_percents]

    bars = plt.bar(range(1, len(cpu_percents) + 1), cpu_percents, color=bar_colors)

    # Add a text label on each bar with the percentage value
    for bar, percent in zip(bars, cpu_percents):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() - 5, f'{percent}%', color='black',
                 fontweight='bold', ha='center')

    plt.ylim([0, 100])
    plt.xticks(ticks=[1, 2, 3, 4], labels=[1, 2, 3, 4])
    plt.xlim(0.5, 4.5)
    plt.xlabel('Platoon Members')
    plt.ylabel('CPU Load')
    plt.title('Live CPU Load')


class LoadDashboard(object):
    def __init__(self,cav_list=None):
        self.states = []
        self.cav_list = cav_list
        self.run_thread = Thread(target=run, args=(self,))
        self.run_thread.start()


def run(dash):
    # Set up the plot
    plt.figure(figsize=(10, 6))
    # Animate the plot with the 'animate' function, interval is set to 1000ms (1 second)
    # ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    ani = FuncAnimation(plt.gcf(), animate, fargs=(dash,), interval=100)
    # Show the plot
    plt.tight_layout()
    plt.show()
