import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import time
from datetime import datetime

# --- Configuration ---
DAQ_DEVICE_NAME = "Dev1"
DIFFERENTIAL_CHANNELS = ["ai0", "ai1", "ai2", "ai3"]
MIN_VOLTAGE = -1.0  # V
MAX_VOLTAGE = 1.0  # V
CALIBRATION_FILE = "calibration_data.json"

# Plotting and Sampling Configuration
SAMPLE_RATE = 100 # Hz (DAQ speed)
SECONDS_TO_COLLECT = 0.1 # Time window for each DAQ read
SAMPLES_PER_POINT = int(SAMPLE_RATE * SECONDS_TO_COLLECT)
PLOT_WINDOW_SECONDS = 60 # How many seconds of data to show on the plot
DATA_POINTS_TO_SHOW = int(PLOT_WINDOW_SECONDS / SECONDS_TO_COLLECT)

# Global data structures
T_INTERPOLATORS = {}
# Data history buffers (time and temperature for plotting)
time_data = np.array([]) 
temp_data = [np.array([]) for _ in DIFFERENTIAL_CHANNELS] 
start_time = time.time() # To track relative time

# --- Cold Junction Compensation Constants & Functions ---
# Type K Thermocouple (NIST ITS-90) EMF Polynomial (Segment 1: 0°C to 500°C)
K_COEFFS_0_500C_uV = [
    0.000000000000, 40.13454479366, 0.0469038230283, -0.00010476856479,
    0.00000021648037042, -0.00000000028882436759, 0.00000000000017168931136
]

def temp_to_emf_k_nist(temp_c):
    """Calculates Type K thermocouple EMF (in V) for a given temperature (°C)."""
    if temp_c < 0.0: temp_c = 0.0
    emf_uV = sum(coeff * (temp_c ** i) for i, coeff in enumerate(K_COEFFS_0_500C_uV))
    return emf_uV / 1000000.0

def get_cold_junction_temp():
    """Prompts the user for the Cold Junction Temperature."""
    while True:
        try:
            cjt_input = input("Enter the **Cold Junction Temperature** (in degrees C): ")
            cjt = float(cjt_input)
            print(f"Cold Junction Temperature set to: {cjt}°C.")
            return cjt
        except ValueError:
            print("Invalid input. Please enter a numerical value for the temperature.")

def load_calibration_curve():
    """Loads calibration data and creates a temperature interpolation function for each channel."""
    global T_INTERPOLATORS
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)

        temp_points = np.array([p["reference_temp_c"] for p in data["data_points"]])
        
        for i in range(len(DIFFERENTIAL_CHANNELS)):
            channel_key = f"Channel_{i}"
            
            voltage_points = np.array([
                p["corrected_voltages_mean"][channel_key] 
                for p in data["data_points"]
            ])

            if len(temp_points) < 2:
                print(f"❌ ERROR: Calibration data for {channel_key} must contain at least 2 points.")
                exit()

            T_INTERPOLATORS[channel_key] = interp1d(
                voltage_points, temp_points, 
                kind='linear', 
                fill_value="extrapolate"
            )
        print(f"✅ Calibration curves loaded for {len(DIFFERENTIAL_CHANNELS)} channels.")

    except FileNotFoundError:
        print(f"❌ ERROR: Calibration file '{CALIBRATION_FILE}' not found. Cannot proceed.")
        exit()
    except Exception as e:
        print(f"❌ ERROR loading calibration data: {e}")
        exit()

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
for i in range(len(DIFFERENTIAL_CHANNELS)):
    line, = ax.plot(time_data, temp_data[i], label=f'Channel {i}', linewidth=2)
    lines.append(line)

ax.set_title("Real-Time Temperature Monitoring (CJC Applied)")
ax.set_xlabel(f"Time since start (s) - Showing last {PLOT_WINDOW_SECONDS}s")
ax.set_ylabel("Temperature (°C)")
ax.grid(True)
ax.legend()


# --- Main Data Collection and Update Function ---
def update_plot(frame, task, cjc_emf_v):
    """
    Function called repeatedly by FuncAnimation to update the plot data.
    """
    global time_data, temp_data
    current_time = time.time() - start_time
    
    try:
        # 1. Read Raw Voltage
        samples = task.read(number_of_samples_per_channel=SAMPLES_PER_POINT)
        raw_mean_voltages = np.mean(np.array(samples), axis=1)

        # Temporary list for new temperatures
        new_temps = []

        # 2. Process Channels
        for i in range(len(DIFFERENTIAL_CHANNELS)):
            channel_key = f"Channel_{i}"
            
            # Apply Cold Junction Compensation
            adjusted_voltage = raw_mean_voltages[i] + cjc_emf_v
            
            # Convert Adjusted Voltage to Temperature
            interpolator = T_INTERPOLATORS.get(channel_key)
            temperature = interpolator(adjusted_voltage) if interpolator else np.nan
            new_temps.append(temperature)

        # 3. Append Data
        new_time_point = current_time
        time_data = np.append(time_data, new_time_point)
        
        # Trim historical data
        if len(time_data) > DATA_POINTS_TO_SHOW:
            time_data = time_data[-DATA_POINTS_TO_SHOW:]
            
        # 4. Update plot data and trim
        for i, temp_array in enumerate(temp_data):
            temp_data[i] = np.append(temp_array, new_temps[i])
            if len(temp_data[i]) > DATA_POINTS_TO_SHOW:
                temp_data[i] = temp_data[i][-DATA_POINTS_TO_SHOW:]
            
            lines[i].set_data(time_data, temp_data[i])

        # 5. Auto-scale X-axis (time)
        if len(time_data) > 1:
            ax.set_xlim(time_data[0], time_data[-1])
        
        # 6. Auto-scale Y-axis (temperature) based on visible data
        all_visible_temps = np.concatenate([arr for arr in temp_data if arr.size > 0])
        if all_visible_temps.size > 0:
            min_temp = np.min(all_visible_temps)
            max_temp = np.max(all_visible_temps)
            y_range = max_temp - min_temp
            # Set Y-limits with a 5% buffer on both sides
            ax.set_ylim(min_temp - y_range * 0.1, max_temp + y_range * 0.1)

        return lines # Return the lines objects that were modified

    except nidaqmx.errors.DaqError as e:
        print(f"\n--- DAQmx Error Occurred during read: {e} ---")
        return lines
    except Exception as e:
        print(f"\n--- General Error: {e} ---")
        return lines


# --- Main Execution Block ---
if __name__ == "__main__":
    load_calibration_curve()
    
    # Get CJT and calculate compensation EMF
    cold_junction_temp = get_cold_junction_temp()
    cjc_emf_v = temp_to_emf_k_nist(cold_junction_temp)
    print(f"CJC EMF (E_cold): {cjc_emf_v:+.9f} V. Starting monitoring...")

    daq_task = None
    try:
        # 1. Setup DAQ Task (Done once outside the animation loop)
        daq_task = nidaqmx.Task()
        for channel_index, chan_name in enumerate(DIFFERENTIAL_CHANNELS):
            physical_channel = f"{DAQ_DEVICE_NAME}/{chan_name}"
            daq_task.ai_channels.add_ai_voltage_chan(
                physical_channel=physical_channel,
                name_to_assign_to_channel=f"Channel_{channel_index}",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=MIN_VOLTAGE,
                max_val=MAX_VOLTAGE
            )
        daq_task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=SAMPLES_PER_POINT
        )
        
        # 2. Start Animation
        # interval is in milliseconds. 1000ms / 2 = 500ms (0.5s) update rate.
        update_interval_ms = int(SECONDS_TO_COLLECT * 1000)
        
        ani = FuncAnimation(
            fig, 
            update_plot, 
            fargs=(daq_task, cjc_emf_v), 
            interval=update_interval_ms, 
            blit=False, 
            cache_frame_data=False # <-- ADD THIS LINE
        )
        
        plt.show() # Blocking call that runs the animation loop

    except nidaqmx.errors.DaqError as e:
        print("\n--- DAQmx Initialization Error Occurred ---")
        print(f"Error Details: {e}")
    except Exception as e:
        print(f"\n--- Unhandled Error: {e} ---")
    finally:
        if daq_task:
            daq_task.close()
        print("\nMonitoring script terminated.")