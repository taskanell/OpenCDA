import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Read data from CSV and parse 'time' as datetime
data = pd.read_csv("gpu_usage.csv", parse_dates=['time'])

# Calculate usage divided by 24564 and rounded up to the nearest integer
data['depaoli usage'] = np.ceil(data['depaoli'] / 24564)
data['mascolini usage'] = np.ceil(data['mascolini'] / 24564)
data['depaoli usage'] = data['depaoli']
data['mascolini usage'] = data['mascolini']

# Plot configuration
fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the GPU usage
ax.plot(data['time'], data['depaoli usage'], label='Depaoli Usage', linestyle='-', linewidth=2, color='green')
ax.plot(data['time'], data['mascolini usage'], label='mascolini Usage', linestyle='-', linewidth=2, color='red')

# Formatting the x-axis to show dates only
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Setting integer locator for y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Setting labels and title
ax.set_xlabel('Time [days]', fontsize=14)
ax.set_ylabel('GPU Usage (rounded up)', fontsize=14)
ax.set_title('GPU Usage Over Time for Depaoli and mascolini', fontsize=16)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('gpu_usage_depaoli_mascolini.png', dpi=320)
plt.show()
