import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --- Configuration ---
DAQ_DEVICE_NAME = "Dev1"
DIFFERENTIAL_CHANNELS = ["ai0", "ai1", "ai2", "ai3"]
MIN_VOLTAGE = -1.0  # V
MAX_VOLTAGE = 1.0  # V
CALIBRATION_POINTS = [0, 20, 100]  # known reference temperatures
OUTPUT_FILENAME = "calibration_data.json"

# Sampling configuration
SAMPLE_RATE = 100  # Hz
SECONDS_TO_COLLECT = 1  # collect 1 second per calibration point
SAMPLES_PER_POINT = SAMPLE_RATE * SECONDS_TO_COLLECT

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
        print(f"WARNING: Cold Junction Temp {temp_c}°C is outside the polynomial range (0-500°C). Using boundary values.")
        if temp_c < 0.0: temp_c = 0.0
        
    emf_uV = 0.0
    for i, coeff in enumerate(K_COEFFS_0_500C_uV):
        emf_uV += coeff * (temp_c ** i)
        
    # Convert microvolts (uV) to Volts (V)
    emf_V = emf_uV / 1000000.0
    return emf_V


# --- Placeholder for External Calibration Control ---
def read_external_calibration_value():
    """
    Replace this with your actual logic for reading or confirming the calibration source.
    For now, it just waits for user confirmation.
    """
    input(">>> PRESS ENTER to confirm the calibration source is set to the NEXT value. ")
    print("Reading measurement...")
    return


# --- Cold Junction Temperature Function ---
def get_cold_junction_temp():
    """Prompts the user for the Cold Junction Temperature."""
    while True:
        try:
            cjt_input = input("\nEnter the **Cold Junction Temperature** (in degrees C): ")
            cjt = float(cjt_input)
            print(f"Cold Junction Temperature set to: {cjt}°C.")
            return cjt
        except ValueError:
            print("Invalid input. Please enter a numerical value for the temperature.")


# --- Main Calibration Function ---
def perform_calibration_and_save():
    """Performs multi-point calibration, saves averaged data, and plots voltage vs. temperature."""
    cold_junction_temp = get_cold_junction_temp()
    
    # Calculate the Cold Junction Compensation (CJC) EMF using the NIST polynomial
    cjc_emf_v = temp_to_emf_k_nist(cold_junction_temp)

    print(f"Calculated Cold Junction Compensation EMF (Type K, {cold_junction_temp}°C): {cjc_emf_v:+.9f} V (NIST)")

    calibration_data = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": DAQ_DEVICE_NAME,
            "range": f"{MIN_VOLTAGE}V to {MAX_VOLTAGE}V",
            "cold_junction_temp_C": cold_junction_temp,
            "cjc_emf_V": cjc_emf_v, # Store the compensation value
            "cjc_polynomial_source": "NIST ITS-90 Type K (0-500C segment)",
            "channels": {f"Channel_{i}": f"{DAQ_DEVICE_NAME}/{ch}" for i, ch in enumerate(DIFFERENTIAL_CHANNELS)},
            "calibration_units": "C",
            "sample_rate_Hz": SAMPLE_RATE,
            "collection_time_s": SECONDS_TO_COLLECT
        },
        "data_points": []
    }

    print(f"\n--- Starting {len(CALIBRATION_POINTS)}-Point Calibration ---")

    try:
        with nidaqmx.Task() as task:
            # Configure channels
            for channel_index, chan_name in enumerate(DIFFERENTIAL_CHANNELS):
                physical_channel = f"{DAQ_DEVICE_NAME}/{chan_name}"
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel=physical_channel,
                    name_to_assign_to_channel=f"Channel_{channel_index}",
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=MIN_VOLTAGE,
                    max_val=MAX_VOLTAGE
                )

            # Timing configuration
            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=SAMPLES_PER_POINT
            )

            print(f"DAQmx Task created with {len(DIFFERENTIAL_CHANNELS)} differential channels.")
            print(f"Collecting {SECONDS_TO_COLLECT} second(s) ({SAMPLES_PER_POINT} samples) per calibration point.\n")

            # Collect data
            for cal_value in CALIBRATION_POINTS:
                print(f"\nSet calibration source to: {cal_value} {calibration_data['metadata']['calibration_units']}...")
                read_external_calibration_value()

                samples = task.read(number_of_samples_per_channel=SAMPLES_PER_POINT)
                samples = np.array(samples)
                mean_voltages = np.mean(samples, axis=1)
                
                # --- COLD JUNCTION COMPENSATION (CJC) ---
                # E_hot = E_measured + E_cold (cjc_emf_v)
                corrected_mean_voltages = mean_voltages + cjc_emf_v 
                # ----------------------------------------

                point_data = {
                    "reference_temp_c": cal_value,
                    "measured_voltages_raw_mean": { 
                        f"Channel_{i}": float(v) for i, v in enumerate(mean_voltages)
                    },
                    "corrected_voltages_mean": { 
                        f"Channel_{i}": float(v) for i, v in enumerate(corrected_mean_voltages)
                    }
                }

                print(f"| Ref Temp Value: {cal_value:<5}|", end=" ")
                for i, voltage in enumerate(corrected_mean_voltages):
                    print(f"Channel_{i} (Corr): {voltage:+.6f} V |", end=" ")
                calibration_data["data_points"].append(point_data)

        # Save JSON 
        with open(OUTPUT_FILENAME, "w") as f:
            json.dump(calibration_data, f, indent=4)

        print(f"\n\n✅ Calibration complete. Data saved to: {OUTPUT_FILENAME}")

        # --- Generate Plot (Uses corrected voltage) ---
        temps = [p["reference_temp_c"] for p in calibration_data["data_points"]]
        plt.figure(figsize=(8, 6))
        for i in range(len(DIFFERENTIAL_CHANNELS)):
            voltages = [p["corrected_voltages_mean"][f"Channel_{i}"] for p in calibration_data["data_points"]]
            plt.plot(voltages, temps, marker='o', label=f"Channel_{i} (CJC Corrected)")

        plt.xlabel("CJC Corrected Measured Voltage (V)")
        plt.ylabel("Reference Temperature (°C)")
        plt.title(f"Calibration Curve: CJC Corrected Voltage vs Temperature (Type K, CJT={cold_junction_temp}°C)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except nidaqmx.errors.DaqError as e:
        print("\n--- DAQmx Error Occurred ---")
        print(f"Error Details: {e}")
        print("----------------------------\n")
    except Exception as e:
        print("\n--- General Error Occurred ---")
        print(f"An unexpected error occurred: {e}")
        print("------------------------------\n")


# --- Run Script ---
if __name__ == "__main__":
    perform_calibration_and_save()