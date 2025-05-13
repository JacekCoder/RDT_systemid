import numpy as np 
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import os

# Motor constants
POLE_PAIRS = 4         # Number of pole pairs
GEAR_RATIO = 22.67     # Gear ratio
SAMPLE_INTERVAL = 0.005  # Assumed log refresh interval (5 ms)
CURRENT_SCALE = 0.01   # Scaling factor for current
FREQ_TO_SPEED_SCALE = 20  # Scaling factor to convert frequency
RESOLUTION = (120/(8*22.67))/180*np.pi # Resolution of the hall sensor

def calculate_speed_from_freq(freq, pole_pairs=POLE_PAIRS, gear_ratio=GEAR_RATIO):
    """

    Args:
        freq (float): commutation frequency in Hz
        pole_pairs (int, optional): number of pole pairs. Defaults to POLE_PAIRS.
        gear_ratio (float, optional): gear ratio. Defaults to GEAR_RATIO.

    Returns:
        angular speed: angular speed in rad/s
    """
    mechanical_rps = (freq * 20) / 60.0 
    mechanical_rps /= (2 * pole_pairs * gear_ratio) 
    speed_rad_s = mechanical_rps * 2 * np.pi

    return speed_rad_s

def calculate_speed_from_abs_pos(abs_pos,timestep,resolution,low_pass_filter=True,lp_alpha=0.05):
    """

    Args:
        abs_pos (int): absolute position measured by hall sensor
        timestep (float): time interval between two measurements
        resolution (float): resolution of the hall sensor
        low_pass_filter (bool, optional): using low pass fileter to filter the signal or not. Defaults to True.
        lp_alpha (float, optional): alpha of low pass filter (first order IIR). Defaults to 0.05.

    Returns:
        angular speed: angular speed in rad/s
    """
    speed = np.zeros_like(abs_pos)
    if low_pass_filter:
        abs_pos_filtered = np.zeros_like(abs_pos)
        abs_pos_filtered[0] = abs_pos[0]
        for i in range(1,abs_pos.shape[0]):
            abs_pos_filtered[i] = lp_alpha*abs_pos[i] + (1-lp_alpha)*abs_pos_filtered[i-1]
        speed[1:] = np.diff(abs_pos_filtered) / timestep
    else:
        speed[1:] = np.diff(abs_pos) / timestep
    speed[0] = speed[1]
    return speed*resolution

def parse_can_log(file_path):
    """

    Args:
        file_path (str): file path of raw can bus log data by canalyst II

    Returns:
        df(Pandas datagframe): parsed data
    """
    pattern = re.compile(
        r"^Timestamp:\s*([\d\.]+)\s+ID:\s*(\d+)\s+S\s+Rx\s+DL:\s*(\d+)\s+([0-9A-Fa-f\s]+)\s+Channel:\s*(\d+)$"
    )
    
    data_records = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = pattern.match(line)
            if not match:
                continue

            # Extract fields (using group(1) to group(5))
            timestamp_str = match.group(1)
            can_id = match.group(2)
            data_bytes_str = match.group(4).strip()

            try:
                timestamp = float(timestamp_str)
                byte_list = data_bytes_str.split()
                data_bytes = [int(b, 16) for b in byte_list]
            except ValueError:
                continue

            # Parse data based on CAN ID
            if can_id == '181':
                if len(data_bytes) >= 7:
                    motor_freq = (data_bytes[1] << 8) | data_bytes[0]
                    if motor_freq & 0x8000:  # Check 16-bit sign bit
                        motor_freq -= 0x10000
                    motor_pos = (data_bytes[5] << 24) | (data_bytes[4] << 16) | (data_bytes[3] << 8) | data_bytes[2]
                    if motor_pos & 0x80000000:  # Check if the sign bit is set
                        motor_pos -= 0x100000000  # Convert to signed 32-bit integer
                    data_records.append([timestamp, "Commutation_freq", motor_freq])
                    data_records.append([timestamp, "Motor_position", motor_pos])
                    
            elif can_id == '281':
                if len(data_bytes) >= 6:
                    motor_current = (data_bytes[1] << 8) | data_bytes[0]
                    pwm = (data_bytes[3] << 8) | data_bytes[2]
                    if pwm & 0x8000:  # Check if the sign bit is set for 16-bit signed value
                        pwm -= 0x10000  # Convert to signed 16-bit integer
                    voltage = (data_bytes[5] << 8) | data_bytes[4]
                    data_records.append([timestamp, "Phase_current", motor_current])
                    data_records.append([timestamp, "PWM", pwm])
                    data_records.append([timestamp, "Voltage", voltage])
            elif can_id == '601':
                if len(data_bytes) >= 5:
                    ctrl_command = data_bytes[4:6]
                    ctrl_value = (ctrl_command[1] << 8) | ctrl_command[0]
                    if ctrl_value & 0x8000:
                        ctrl_value -= 0x10000
                    data_records.append([timestamp, "Ctrl_command", ctrl_value])
    
    df = pd.DataFrame(data_records, columns=["Timestamp", "Parameter", "Value"])
    return df

def main(file_path):
    df = parse_can_log(file_path)
    if df.empty:
        print("❌ No matching data found. Check data file or regex.")
        return
    
    df_freq = df[df.Parameter == "Commutation_freq"].reset_index(drop=True)
    time1 = df_freq["Timestamp"].to_numpy()
    time1_interval = np.diff(time1)
    speed = calculate_speed_from_freq(df_freq["Value"].to_numpy()*0.1)
    print(speed.shape)
    
    df_pos = df[df.Parameter == "Motor_position"].reset_index(drop=True)
    pos = df_pos["Value"].to_numpy()
    
    df_current = df[df.Parameter == "Phase_current"].reset_index(drop=True)
    time2 = df_current["Timestamp"].to_numpy()
    current = df_current["Value"].to_numpy() * 0.01
    time2_interval = np.diff(time2)
    print(current.shape)
    
    df_PWM = df[df.Parameter == "PWM"].reset_index(drop=True)
    pwm = df_PWM["Value"].to_numpy() * 0.1
    
    df_voltage = df[df.Parameter == "Voltage"].reset_index(drop=True)
    voltage = df_voltage["Value"].to_numpy() * 0.1
    
    df_ctrl = df[df.Parameter == "Ctrl_command"].reset_index(drop=True)
    ctrl = calculate_speed_from_freq(df_ctrl["Value"].to_numpy()*0.1)
    time_ctrl = df_ctrl["Timestamp"].to_numpy()
    time_ctrl_interval = np.diff(time_ctrl)
    
    if speed.shape[0] != current.shape[0]:
        if speed.shape[0] > current.shape[0]:
            speed = speed[:current.shape[0]]
            pos = pos[:current.shape[0]]
        else:
            current = current[:speed.shape[0]]
            pwm = pwm[:speed.shape[0]]
            voltage = voltage[:speed.shape[0]]
    # check speed shape and current shapre are equal or not
    assert speed.shape[0] == current.shape[0], "Speed and current data length mismatch"
    
    speed2 = calculate_speed_from_abs_pos(pos, SAMPLE_INTERVAL,resolution=RESOLUTION,low_pass_filter=True,lp_alpha=0.05)
    
    current_direction = np.sign(pwm)
    current_direction[current_direction == 0] = 1
    current = current * current_direction
    
    # check time interval if there is any missing data
    assert np.max(np.abs(time1_interval-SAMPLE_INTERVAL)) < SAMPLE_INTERVAL/5, "Missing data in logged file"
    assert np.max(np.abs(time2_interval-SAMPLE_INTERVAL)) < SAMPLE_INTERVAL/5, "Missing data in logged file"
    
    if_plot = False
    
    if if_plot:
        # check time1 interval
        plt.figure(figsize=(10, 5))
        plt.plot(time1[1:], time1_interval,"*")
        
        
        # # check time2 interval
        plt.figure(figsize=(10, 5))
        plt.plot(time2[1:], time2_interval,"*")
            
        
        plt.figure(figsize=(10, 5))
        plt.plot(current,label="Current")
        plt.plot(speed,label="Speed")
        plt.plot(speed2,label="Speed2")
        plt.legend()
        
        plt.figure(figsize=(10, 5))
        plt.plot(pwm,"o",label="PWM")
        plt.plot(pos,"-o",label="Position")
        plt.legend()
        
        # check time_ctrl interval
        plt.figure(figsize=(10, 5))
        plt.plot(time_ctrl[1:], time_ctrl_interval,"*")
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_ctrl, ctrl,label="Ctrl")
        plt.show()
    
    # save data to csv: time, speed, pos, current, pwm, voltage
    time = np.arange(0,speed.shape[0])*SAMPLE_INTERVAL
    datatosave = np.column_stack((time, speed, pos, current, pwm, voltage))
    datatocsv = pd.DataFrame(datatosave, columns=["Time", "Speed", "Position", "Current", "PWM", "Voltage"])
    save_path = file_path.replace("raw", "sysid").replace(".txt", ".csv")
    datatocsv.to_csv(save_path, index=False)
    
    # save ctrl data to csv
    datatocsv_ctrl = np.column_stack((time_ctrl, ctrl))
    datatocsv_ctrl = pd.DataFrame(datatocsv_ctrl, columns=["Time", "Ctrl"])
    save_path = file_path.replace("raw", "sysid").replace(".txt", "_ctrl.csv")
    datatocsv_ctrl.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    batch_process = False
    
    if not batch_process:
        file_path = "sysid_data/DRBA/motor/raw/canalystII/set3_has_1.txt"
        main(file_path)
    
    if batch_process:
        # batch process all the files in the folder
        folder_path = "sysid_data/DRBA/motor/raw/canalystII/"
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                main(file_path)