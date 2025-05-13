import numpy as np
import pandas as pd
import re
# import ace_tools_open as tools
import matplotlib.pyplot as plt

# Improved function to handle inconsistencies in CAN log file format
def parse_can_log_refined(file_path):
    data_records = []

    # Regular expression pattern to match CAN log lines
    pattern = re.compile(r"([\d\.]+);\s*RX;\s*0x([0-9A-Fa-f]+);\s*(\d+);\s*((?:0x[0-9A-Fa-f]+;?)+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                timestamp, can_id, dlc, data_bytes = match.groups()
                # Keep can_id as a string with "0x" notation
                timestamp = float(timestamp)  # Convert timestamp to float
                dlc = int(dlc)  # Convert DLC to int
                can_id = "0x" + can_id  # Add "0x" prefix to CAN ID
                # Cleaning up the data bytes format
                byte_list = re.findall(r"0x[0-9A-Fa-f]+", data_bytes)  # Extract valid hex values
                try:
                    data_bytes = [int(byte, 16) for byte in byte_list]  # Convert hex bytes to list
                except ValueError:
                    continue  # Skip lines with invalid formatting
                # Extract relevant fields based on CAN ID
                if can_id in ['0x181', '0x182']:  # Left (0x181) and Right (0x182) motor data
                    if can_id == '0x181':
                        motor_side = "Left"
                    else:
                        motor_side = "Right"
                    if len(data_bytes) >= 7:
                        motor_freq = (data_bytes[1] << 8) | data_bytes[0]
                        if motor_freq & 0x8000:  # Check if the sign bit is set for 16-bit signed value
                            motor_freq -= 0x10000  # Convert to signed 16-bit integer
                        motor_pos = (data_bytes[5] << 24) | (data_bytes[4] << 16) | (data_bytes[3] << 8) | data_bytes[2]
                        if motor_pos & 0x80000000:  # Check if the sign bit is set
                            motor_pos -= 0x100000000  # Convert to signed 32-bit integer
                        data_records.append([timestamp, motor_side, "Commutation_freq", motor_freq])
                        data_records.append([timestamp, motor_side, "Absolute_pos", motor_pos])

                elif can_id in ['0x281', '0x282']:  # Left (0x281) and Right (0x282) motor status
                    if can_id == '0x281':
                        motor_side = "Left"
                    else:
                        motor_side = "Right"
                    if len(data_bytes) >= 6:
                        motor_current = (data_bytes[1] << 8) | data_bytes[0]
                        pwm = (data_bytes[3] << 8) | data_bytes[2]
                        if pwm & 0x8000:  # Check if the sign bit is set for 16-bit signed value
                            pwm -= 0x10000  # Convert to signed 16-bit integer
                        voltage = (data_bytes[5] << 8) | data_bytes[4]
                        data_records.append([timestamp, motor_side, "Phase_current", motor_current])
                        data_records.append([timestamp, motor_side, "PWM", pwm])
                        data_records.append([timestamp, motor_side, "Voltage", voltage])

                elif can_id in ['0x601', '0x602']:  # Control commands
                    if can_id == '0x601':
                        motor_side = "Left"
                    else:
                        motor_side = "Right"
                    if len(data_bytes) >= 5:
                        control_command = data_bytes[4:6]  # Reverse the order of the last bytes
                        control_value = (control_command[1] << 8) | control_command[0]
                        if control_value & 0x8000:  # Check if the sign bit is set for 16-bit signed value
                            control_value -= 0x10000  # Convert to signed 16-bit integer
                        data_records.append([timestamp, motor_side, "Ctrl_input", control_value])

    # Convert to DataFrame for better visualization
    df = pd.DataFrame(data_records, columns=["Timestamp", "Side", "Parameter", "Value"])
    
    return df

log_file_path = "sysid_data/DRBA/motor/raw/motor_test1.txt"
# Parse CAN log file again with the refined function
df_can_log_refined = parse_can_log_refined(log_file_path)


# Separate data into two DataFrames by side (excluding the "Side" column)
df_left = df_can_log_refined[df_can_log_refined["Side"] == "Left"].drop(columns=["Side"]).reset_index(drop=True)
df_right = df_can_log_refined[df_can_log_refined["Side"] == "Right"].drop(columns=["Side"]).reset_index(drop=True)

# Separate "Ctrl_input" from the other parameters for both sides
df_left_ctrl = df_left[df_left["Parameter"] == "Ctrl_input"].reset_index(drop=True)
df_left_feedback = df_left[df_left["Parameter"] != "Ctrl_input"].reset_index(drop=True)

df_right_ctrl = df_right[df_right["Parameter"] == "Ctrl_input"].reset_index(drop=True)
df_right_feedback = df_right[df_right["Parameter"] != "Ctrl_input"].reset_index(drop=True)

df_left_freq = df_left_feedback[df_left_feedback["Parameter"] == "Commutation_freq"].reset_index(drop=True)
df_left_pos = df_left_feedback[df_left_feedback["Parameter"] == "Absolute_pos"].reset_index(drop=True)
df_left_current = df_left_feedback[df_left_feedback["Parameter"] == "Phase_current"].reset_index(drop=True)
df_left_voltage = df_left_feedback[df_left_feedback["Parameter"] == "Voltage"].reset_index(drop=True)
df_left_pwm = df_left_feedback[df_left_feedback["Parameter"] == "PWM"].reset_index(drop=True)

timestamp = df_left_freq["Timestamp"].to_numpy()
time_interval = np.diff(timestamp)
print(time_interval.shape)
print(timestamp.shape)
#plot left motor feedback data in subplots
plt.figure()
plt.plot(timestamp[1:], time_interval,'*')
plt.title("Data logging missing check")
plt.xlabel("Timestamp")
plt.ylabel("Time interval")
plt.show()

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(df_left_freq["Timestamp"], df_left_freq["Value"]*0.1)
plt.title("Left Motor Commutation Frequency")
plt.xlabel("Timestamp")
plt.ylabel("Frequency (Hz)")

plt.subplot(2, 3, 2)
plt.plot(df_left_pos["Timestamp"], df_left_pos["Value"])
plt.title("Left Motor Absolute Position")
plt.xlabel("Timestamp")
plt.ylabel("Position")

plt.subplot(2, 3, 3)
plt.plot(df_left_current["Timestamp"], df_left_current["Value"]*0.01)
plt.title("Left Motor Phase Current")
plt.xlabel("Timestamp")
plt.ylabel("Current (A)")

plt.subplot(2, 3, 4)
plt.plot(df_left_voltage["Timestamp"], df_left_voltage["Value"]*0.1)
plt.title("Left Motor Voltage")
plt.xlabel("Timestamp")
plt.ylabel("Voltage (V)")

plt.subplot(2, 3, 5)
plt.plot(df_left_pwm["Timestamp"], df_left_pwm["Value"]*0.1)
plt.title("Left Motor PWM")
plt.xlabel("Timestamp")
plt.ylabel("PWM %")

plt.subplot(2, 3, 6)
plt.plot(df_left_ctrl["Timestamp"], df_left_ctrl["Value"]*0.1)
plt.title("Left Motor Control Input")
plt.xlabel("Timestamp")
plt.ylabel("Control Input")

plt.tight_layout()
plt.show()