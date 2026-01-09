import pynvml
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime

# Initialize NVML
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
except Exception as e:
    print(f"Failed to initialize NVML: {e}")
    exit()

# Arrays for your ML training data
time_data = []
temp_data = []
mem_data = []
util_data = []

print("GPU Sensor connected! Reading data...")
print(f"Device: {name}")

def update(i):
    try:
        # Get current temp as a float for precision
        temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        mem = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024 * 1024)  # in MB
        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)  # in %
        current_time = datetime.now().strftime('%H:%M:%S')

        # Update data arrays
        time_data.append(current_time)
        temp_data.append(temp)
        mem_data.append(mem)
        util_data.append(util)

        # Keep a window of the last 60 readings for the display
        if len(time_data) > 60:
            time_data.pop(0)
            temp_data.pop(0)
            mem_data.pop(0)
            util_data.pop(0)

        # Clear and Redraw
        plt.cla()
        ax1.cla(); ax2.cla(); ax3.cla()

        #Subplot 1: Temperature
        ax1.plot(time_data, temp_data, marker='o', color='red', linestyle='-', linewidth=2)
        ax1.set_title(f"GPU: {name} | Current Temp: {temp}°C")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_ylim(min(temp_data)-2, max(temp_data)+2)
        ax1.tick_params(labelbottom=False, bottom=False)
        ax1.grid(True, alpha=0.3)

        #Subplot 2: Memory Usage
        ax2.plot(time_data, mem_data, marker='o', color='blue', linestyle='-', linewidth=2)
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.tick_params(labelbottom=False, bottom=False)
        ax2.grid(True, alpha=0.3)

        #Subplot 3: GPU Utilization
        ax3.plot(time_data, util_data, marker='o', color='green', linestyle='-', linewidth=2)
        ax3.set_ylabel("GPU Utilization (%)")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        #Improve layout
        plt.xlabel("Time")
        plt.xticks(rotation=45)
        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()

    except Exception as e:
        print(f"Error during update: {e}")

# Create animation: interval=5000ms (5 seconds)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8))
ani = FuncAnimation(fig, update, interval=5000, cache_frame_data=False)

try:
    plt.show()
finally:
    # Ensure NVML shuts down when the window is closed
    pynvml.nvmlShutdown()
    print("NVML Shutdown safely.")
