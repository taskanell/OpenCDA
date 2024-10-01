import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Read data from CSV and parse 'time' as datetime
data = pd.read_csv("gpu_usage.csv", parse_dates=['time'])

# GPU memory usage columns
gpu_columns = ['GPU0 memory used', 'GPU1 memory used', 'GPU2 memory used', 'GPU3 memory used']

# Determine the number of GPUs used at each time point
data['GPUs used'] = data[gpu_columns].apply(lambda x: (x > 1000).sum(), axis=1)

# Plot configuration
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the number of GPUs used over time
ax.plot(data['time'], data['GPUs used'], label='Number of GPUs Used', linestyle='-', linewidth=2, color='blue')

# Formatting the x-axis to show dates only
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Setting labels and title
ax.set_xlabel('Time [days]', fontsize=14)
ax.set_ylabel('Number of GPUs Used', fontsize=14)
ax.set_title('Number of GPUs Used Over Time', fontsize=16)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12)


plt.tight_layout()
plt.savefig('gpu_usage_over_time.png', dpi=320)
plt.show()

