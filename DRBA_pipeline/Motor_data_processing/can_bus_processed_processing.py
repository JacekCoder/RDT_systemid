import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# motor constants
motor_constants = {
    "pole_pairs": 4,
    "gear_ratio": 22.67,
    "hall_sensor_resolution": 57,
    "Kt": 0.72
}

def calculate_speed_from_freq(freq, pole_pairs=motor_constants["pole_pairs"], gear_ratio=motor_constants["gear_ratio"]):
    """
    Calculate the speed of the motor from the commutation frequency.

    Args:
        freq (float): commutation frequency of the motor.
        pole_pairs (int, optional): number of pole pairs. Defaults to motor_constants["pole_pairs"].
        gear_ratio (float, optional): gear ratio of the motor. Defaults to motor_constants["gear_ratio"].
    
    Returns:
        float: speed of the motor.
    """
    return freq*20/(2*pole_pairs*gear_ratio)*2*np.pi/60

def calculate_speed_from_abs_pos(abs_pos,timestep,resolution=motor_constants["hall_sensor_resolution"],low_pass_filter=True,lp_alpha=0.05):
    """
    Calculate the speed from the absolute position of the motor by differentiating the position.
    
    Args:
        abs_pos (float): absolute position of the motor.
        timestep (float): time step between position readings.
        resolution (int, optional): resolution of the hall sensor. Defaults to motor_constants["hall_sensor_resolution"].
        low_pass_filter (bool, optional): apply low pass filter to the speed. Defaults to False.
        lp_alpha (float, optional): alpha value for the low pass filter. Defaults to 0.05.
    Returns:
        float: speed of the motor.
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
    return speed * 2*np.pi / resolution


if __name__ == "__main__":
    file_path = 'sysid_data/DRBA/motor/processed/right_motor.csv'
    df = pd.read_csv(file_path)
    timestep = df["time"][1] - df["time"][0]
    Speed1 = calculate_speed_from_freq(df["Commutation_freq"])
    Speed2 = calculate_speed_from_abs_pos(df["Absolute_pos"],timestep)
    tau = df["Phase_current"] * motor_constants["Kt"]*motor_constants["gear_ratio"]
    # save as sysid data as csv
    sysid_data = np.column_stack((df["time"],Speed1,df["Phase_current"]))
    sysid_data = pd.DataFrame(sysid_data,columns=["time","Speed","Current"])
    sysid_data.to_csv("sysid_data/DRBA/motor/sysid/right_motor_sysid.csv",index=False)
    
    # plot the speed data and torque data
    plt.figure()
    plt.plot(df["time"], -Speed1, label="Speed from Commutation Frequency")
    plt.plot(df["time"], -Speed2, label="Speed from Absolute Position")
    plt.plot(df["time"], tau, label="Torque")
    plt.title("Speed of the Right Motor")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (rad/s)")
    plt.legend()
    plt.show()