"""
=========================
Script description
=========================
This script is used to parse the CAN log file of DRBA using microchip CAN bus analyzer in linux based on mcba_usb.
Author @ rris_Wyf
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Improved function to handle inconsistencies in new CAN log file format
def parse_can_log_refined(file_path):
    data_records = []
    i = 0
    # Regular expression pattern to match CAN log lines
    pattern = re.compile(r"\(([\d\.]+)\)\s+(\S+)\s+(\d+)\s+\[(\d+)\]\s+([\dA-Fa-f\s]+)")
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                timestamp, channel, can_id, bytes_num,data_bytes = match.groups()
                timestamp = float(timestamp)  # Convert timestamp to float
                byte_list = data_bytes.strip().split()  # Extract valid hex values as list
                try:
                    data_bytes = [int(byte, 16) for byte in byte_list]  # Convert hex bytes to list
                except ValueError:
                    continue
                if can_id in ['181', '182']:  # Left (0x181) and Right (0x182) motor data
                    if can_id == '181':
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

                elif can_id in ['281', '282']:  # Left (0x281) and Right (0x282) motor status
                    if can_id == '281':
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

                elif can_id in ['601', '602']:  # Control commands
                    if can_id == '601':
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

if __name__=="__main__":
    log_file_path = "sysid_data/DRBA/motor/raw/motor_data3.txt"
    # Parse CAN log file again with the refined function
    df_can_log_refined = parse_can_log_refined(log_file_path)
    # print(df_can_log_refined)

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

df_right_freq = df_right_feedback[df_right_feedback["Parameter"] == "Commutation_freq"].reset_index(drop=True)
df_right_pos = df_right_feedback[df_right_feedback["Parameter"] == "Absolute_pos"].reset_index(drop=True)
df_right_current = df_right_feedback[df_right_feedback["Parameter"] == "Phase_current"].reset_index(drop=True)
df_right_voltage = df_right_feedback[df_right_feedback["Parameter"] == "Voltage"].reset_index(drop=True)
df_right_pwm = df_right_feedback[df_right_feedback["Parameter"] == "PWM"].reset_index(drop=True)

print(df_left_freq.shape)
print(df_left_pos.shape)
print(df_left_current.shape)
print(df_right_freq.shape)
print(df_right_pos.shape)
print(df_right_current.shape)

timestamp = df_left_freq["Timestamp"].to_numpy()
time_interval = np.diff(timestamp)

timestamp2 = df_left_ctrl["Timestamp"].to_numpy()
time_interval2 = np.diff(timestamp2)

timestamp_r = df_right_freq["Timestamp"].to_numpy()
time_interval_r = np.diff(timestamp_r)

timestamp2_r = df_right_ctrl["Timestamp"].to_numpy()
time_interval2_r = np.diff(timestamp2_r)

ToPlot = True
if ToPlot:

#plot left motor feedback data in subplots
    plt.figure()
    plt.plot(timestamp[1:], time_interval,'*')
    plt.title("Data logg missing check")
    plt.xlabel("Timestamp")
    plt.ylabel("Timeinterval")
    # plt.show()

    plt.figure()
    plt.plot(timestamp2[1:], time_interval2,'*')
    plt.title("Data logg missing check")
    plt.xlabel("Timestamp")
    plt.ylabel("Timeinterval")

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
    # plt.show()

    # Plot right motor feedback data in subplots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(df_right_freq["Timestamp"], df_right_freq["Value"]*0.1)
    plt.title("Right Motor Commutation Frequency")
    plt.xlabel("Timestamp")
    plt.ylabel("Frequency (Hz)")

    plt.subplot(2, 3, 2)
    plt.plot(df_right_pos["Timestamp"], df_right_pos["Value"])
    plt.title("Right Motor Absolute Position")
    plt.xlabel("Timestamp")
    plt.ylabel("Position")

    plt.subplot(2, 3, 3)
    plt.plot(df_right_current["Timestamp"], df_right_current["Value"]*0.01)
    plt.title("Right Motor Phase Current")
    plt.xlabel("Timestamp")
    plt.ylabel("Current (A)")

    plt.subplot(2, 3, 4)
    plt.plot(df_right_voltage["Timestamp"], df_right_voltage["Value"]*0.1)
    plt.title("Right Motor Voltage")
    plt.xlabel("Timestamp")
    plt.ylabel("Voltage (V)")

    plt.subplot(2, 3, 5)
    plt.plot(df_right_pwm["Timestamp"], df_right_pwm["Value"]*0.1)
    plt.title("Right Motor PWM")
    plt.xlabel("Timestamp")
    plt.ylabel("PWM %")

    plt.subplot(2, 3, 6)
    plt.plot(df_right_ctrl["Timestamp"], df_right_ctrl["Value"]*0.1)
    plt.title("Right Motor Control Input")
    plt.xlabel("Timestamp")
    plt.ylabel("Control Input")

    plt.tight_layout()
    plt.show()

# save left motor feedback data into csv file
# time is from zero, time interval is 0.005, shape same as timestamp
time_left = np.arange(0, timestamp.shape[0]*0.005, 0.005)
# header = ["time", "Commutation_freq", "Absolute_pos", "Phase_current", "Voltage", "PWM"]
data = np.column_stack((time_left, df_left_freq["Value"]*0.1, df_left_pos["Value"], df_left_current["Value"]*0.01, df_left_voltage["Value"]*0.1, df_left_pwm["Value"]*0.1))
df_left_motor = pd.DataFrame(data, columns=["time", "Commutation_freq", "Absolute_pos", "Phase_current", "Voltage", "PWM"])
df_left_motor.to_csv("sysid_data/DRBA/motor/processed/left_motor.csv", index=False)

# save right motor feedback data into csv file
# time is from zero, time interval is 0.005, shape same as timestamp
time_right = np.arange(0, timestamp_r.shape[0]*0.005, 0.005)
# header = ["time", "Commutation_freq", "Absolute_pos", "Phase_current", "Voltage", "PWM"]
data = np.column_stack((time_right[:-1], df_right_freq["Value"][:-1]*0.1, df_right_pos["Value"][:-1], df_right_current["Value"]*0.01, df_right_voltage["Value"]*0.1, df_right_pwm["Value"]*0.1))
df_right_motor = pd.DataFrame(data, columns=["time", "Commutation_freq", "Absolute_pos", "Phase_current", "Voltage", "PWM"])
df_right_motor.to_csv("sysid_data/DRBA/motor/processed/right_motor.csv", index=False)

# save left motor control data into csv file
# time is from zero, time interval is 0.065, shape same as timestamp
time_left_ctrl = np.arange(0, timestamp2.shape[0]*0.065, 0.065)
# header = ["time", "Control_input"]
data = np.column_stack((time_left_ctrl, df_left_ctrl["Value"]*0.1))
df_left_ctrl = pd.DataFrame(data, columns=["time", "Control_input"])
df_left_ctrl.to_csv("sysid_data/DRBA/motor/processed/left_motor_ctrl.csv", index=False)

# save right motor control data into csv file
# time is from zero, time interval is 0.065, shape same as timestamp
time_right_ctrl = np.arange(0, timestamp2_r.shape[0]*0.065, 0.065)
# header = ["time", "Control_input"]
data = np.column_stack((time_right_ctrl, df_right_ctrl["Value"]*0.1))
df_right_ctrl = pd.DataFrame(data, columns=["time", "Control_input"])
df_right_ctrl.to_csv("sysid_data/DRBA/motor/processed/right_motor_ctrl.csv", index=False)