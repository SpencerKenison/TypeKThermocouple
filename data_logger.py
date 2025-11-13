import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import json
import time
import numpy as np
import csv
from datetime import datetime
from scipy.interpolate import interp1d
import os 

# --- Configuration ---
DAQ_DEVICE_NAME = "Dev1"
DIFFERENTIAL_CHANNELS = ["ai0", "ai1", "ai2", "ai3"]
MIN_VOLTAGE = -1.0  # V
MAX_VOLTAGE = 1.0  # V
CALIBRATION_FILE = "calibration_data.json"
OUTPUT_FILENAME = "experimental_data.csv" 

# Sampling configuration (Used for each logged point)
SAMPLE_RATE = 100  # Hz
SECONDS_TO_COLLECT = 1  # collect 1 second per data point
SAMPLES_PER_POINT = SAMPLE_RATE * SECONDS_TO_COLLECT

# Global dictionary to store interpolation function for EACH channel
T_INTERPOLATORS = {}


# --- Type K Thermocouple (NIST ITS-90) EMF Polynomial (Segment 1: 0°C to 500°C) ---
# EMF (E) in microvolts (uV) for Temperature (T) in degrees C.
K_COEFFS_0_500C_uV = [
    0.000000000000,
    40.13454479366,
    0.0469038230283,
    -0.00010476856479,
    0.00000021648037042,
    -0.00000000028882436759,
    0.00000000000017168931136
]

def temp_to_emf_k_nist(temp_c):
    """
    Calculates Type K thermocouple EMF (in V) for a given temperature (°C)
    using the NIST ITS-90 polynomial (0°C to 500°C segment).
    """
    if temp_c < 0.0 or temp_c > 500.0:
        if temp_c < 0.0: temp_c = 0.0
        
    emf_uV = 0.0
    for i, coeff in enumerate(K_COEFFS_0_500C_uV):
        emf_uV += coeff * (temp_c ** i)
        
    # Convert microvolts (uV) to Volts (V)
    emf_V = emf_uV / 1000000.0
    return emf_V


# --- Utility Functions ---

def fahrenheit_to_celsius(temp_f):
    """Converts temperature from Fahrenheit to Celsius."""
    # T(°C) = (T(°F) - 32) * 5/9
    return (temp_f - 32.0) * (5.0/9.0)

def celsius_to_fahrenheit(temp_c):
    """Converts temperature from Celsius to Fahrenheit."""
    # T(°F) = T(°C) * 9/5 + 32
    return (temp_c * (9.0/5.0)) + 32.0

def load_calibration_curve():
    """Loads calibration data and creates a temperature interpolation function for each channel."""
    global T_INTERPOLATORS
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            data = json.load(f)

        if not data["data_points"]:
            raise ValueError(f"Calibration file '{CALIBRATION_FILE}' contains no data points.")

        # Extract common temperature points
        temp_points = np.array([p["reference_temp_c"] for p in data["data_points"]])
        
        # Iterate over all configured channels
        for i in range(len(DIFFERENTIAL_CHANNELS)):
            channel_key = f"Channel_{i}"
            
            # Extract corrected voltage points specific to this channel
            try:
                voltage_points = np.array([
                    p["corrected_voltages_mean"][channel_key] 
                    for p in data["data_points"]
                ])
            except KeyError:
                print(f"❌ ERROR: Missing corrected voltage data for {channel_key} in calibration file.")
                raise

            if len(temp_points) < 2:
                raise ValueError(f"Calibration data for {channel_key} must contain at least 2 points.")

            # Create a linear interpolation function: V -> T for this specific channel
            T_INTERPOLATORS[channel_key] = interp1d(
                voltage_points, temp_points, 
                kind='linear', 
                fill_value="extrapolate"
            )
            print(f"✅ Calibration curve loaded for {channel_key}.")

    except FileNotFoundError:
        print(f"❌ ERROR: Calibration file '{CALIBRATION_FILE}' not found. Cannot proceed.")
        exit()
    except Exception as e:
        print(f"❌ ERROR loading calibration data: {e}")
        exit()


def get_user_inputs():
    """Prompts the user for Cold Junction Temp, Rack Height, and Setpoint Temp (F)."""
    while True:
        try:
            cjt_input = input("\nEnter the **Cold Junction Temperature** (in degrees C): ")
            cjt = float(cjt_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for CJT.")

    while True:
        try:
            height_input = input("Enter the **Rack Height** (in meters, m): ")
            height = float(height_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for Rack Height.")
            
    # --- NEW INPUT ---
    while True:
        try:
            setpoint_f_input = input("Enter the **Setpoint Temperature** (in degrees F): ")
            setpoint_f = float(setpoint_f_input)
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value for Setpoint Temperature.")
            
    # Convert and return all values
    setpoint_c = fahrenheit_to_celsius(setpoint_f)
    print(f"Setpoint {setpoint_f}°F converted to {setpoint_c:.2f}°C.")
    
    return cjt, height, setpoint_c


def initialize_csv(filename):
    """Initializes the CSV file with headers only if it doesn't exist."""
    
    file_exists = os.path.exists(filename)
    
    if not file_exists:
        # Define headers (Added Avg_Temp_C column)
        headers = ["Timestamp", "Rack_Height_m", "Setpoint_C", "CJ_Temp_C", "CJ_EMF_V", "Avg_Temp_C"]
        for i in range(len(DIFFERENTIAL_CHANNELS)):
            headers.append(f"Channel_{i}_Raw_V")
            headers.append(f"Channel_{i}_Adjusted_V")
            headers.append(f"Channel_{i}_Temp_C")
            
        # Write headers if file is new 
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"\nCreated new data file: {filename}")
    else:
        print(f"\nAppending data to existing file: {filename}")


# --- Main Logging Function ---

def log_data():
    """Performs data collection, compensation, conversion, and logging."""
    
    output_filename = OUTPUT_FILENAME 
    initialize_csv(output_filename) 

    print("\n--- Starting Data Logging Loop (Press Ctrl+C to stop) ---")

    try:
        with nidaqmx.Task() as task:
            # Configure channels (unchanged)
            for channel_index, chan_name in enumerate(DIFFERENTIAL_CHANNELS):
                physical_channel = f"{DAQ_DEVICE_NAME}/{chan_name}"
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel=physical_channel,
                    name_to_assign_to_channel=f"Channel_{channel_index}",
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=MIN_VOLTAGE,
                    max_val=MAX_VOLTAGE
                )

            # Timing configuration (unchanged)
            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=SAMPLES_PER_POINT
            )

            print(f"DAQmx Task created with {len(DIFFERENTIAL_CHANNELS)} channels.")
            
            while True:
                # 1. Get User Inputs
                cjt, rack_height, setpoint_c = get_user_inputs()
                
                # 2. WAIT FOR USER CONFIRMATION HERE
                input(f"\n>>> Measured data will be logged with CJT={cjt}°C, Height={rack_height} m, and Setpoint={setpoint_c:.2f}°C. PRESS ENTER to confirm and collect data. ")
                print("Collecting data...")
                
                # 3. Calculate Cold Junction EMF (E_cold)
                cjc_emf_v = temp_to_emf_k_nist(cjt)

                # 4. Read Raw Voltage
                samples = task.read(number_of_samples_per_channel=SAMPLES_PER_POINT)
                raw_mean_voltages = np.mean(np.array(samples), axis=1)

                # 5. Process and prepare data row
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Temporary list to hold the individual channel temperatures
                temperatures_c = []
                
                summary_output = [f"--- Logged Data Point ({timestamp}) ---",
                                  f"CJ Temp: {cjt}°C | CJ Adj: {cjc_emf_v:+.6f} V | Height: {rack_height} m | Setpoint: {setpoint_c:.2f} °C"]

                for i in range(len(DIFFERENTIAL_CHANNELS)):
                    channel_key = f"Channel_{i}"
                    
                    # Apply Cold Junction Compensation
                    adjusted_voltage = raw_mean_voltages[i] + cjc_emf_v
                    
                    # Convert Adjusted Voltage to Temperature using THIS channel's interpolator
                    interpolator = T_INTERPOLATORS.get(channel_key)
                    if interpolator:
                        temperature = interpolator(adjusted_voltage)
                    else:
                        temperature = np.nan 

                    # Store for averaging later
                    temperatures_c.append(temperature)

                    # Update summary output
                    summary_output.append(
                        f"{channel_key}: Raw V={raw_mean_voltages[i]:+.6f}, "
                        f"Adj V={adjusted_voltage:+.6f}, "
                        f"Temp={temperature:.2f} °C"
                    )
                
                # 6. Calculate Average Temperature (C)
                valid_temps = [t for t in temperatures_c if not np.isnan(t)]
                if valid_temps:
                    avg_temp_c = np.mean(valid_temps)
                    avg_temp_f = celsius_to_fahrenheit(avg_temp_c)
                else:
                    avg_temp_c = np.nan
                    avg_temp_f = np.nan
                
                # 7. Construct Log Row (Added Avg_Temp_C)
                # Log Row Order: Timestamp, Rack_Height_m, Setpoint_C, CJ_Temp_C, CJ_EMF_V, Avg_Temp_C, ...
                log_row = [timestamp, rack_height, setpoint_c, cjt, cjc_emf_v, avg_temp_c]
                
                # Add individual channel data to the log row
                for i in range(len(DIFFERENTIAL_CHANNELS)):
                    adjusted_voltage = raw_mean_voltages[i] + cjc_emf_v
                    log_row.append(raw_mean_voltages[i])
                    log_row.append(adjusted_voltage)
                    log_row.append(temperatures_c[i]) # Use the calculated temperature

                # 8. Log Data to CSV 
                with open(output_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(log_row)
                    
                # 9. Print Summary
                print("\n".join(summary_output))
                print("--------------------------------------")
                
                # Print Averages
                print("\n" + "="*35)
                print(f"| Average Temperature: {avg_temp_c:.2f} °C / {avg_temp_f:.2f} °F |")
                print("="*35)


    except nidaqmx.errors.DaqError as e:
        print("\n--- DAQmx Error Occurred ---")
        print(f"Error Details: {e}")
        print("----------------------------\n")
    except KeyboardInterrupt:
        print("\n\n🛑 Logging stopped by user.")
    except Exception as e:
        print("\n--- General Error Occurred ---")
        print(f"An unexpected error occurred: {e}")
        print("------------------------------\n")


# --- Run Script ---
if __name__ == "__main__":
    load_calibration_curve()
    log_data()