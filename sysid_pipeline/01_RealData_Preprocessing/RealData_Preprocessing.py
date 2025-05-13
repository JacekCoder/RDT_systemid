"""
=========================
Script description
=========================
This script is used to preprocess the real data of joint positions and torques collected from the real robot (in .csv format).
Different filters can be applied (frequency domain needs to be further developed and validated).
Joint velocties and accelerations are calcualted accordingly.
Processed data is saved in a csv file in the current directory.
Author @ rris-Wyf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft,fftfreq,ifft

def avg_low_pass_filter(data, window_size):
    """
    Apply a low pass filter to the data using a moving average filter
    
    Args:
        data (np.array): data to be filtered
        window_size (int): size of the window for the moving average filter
        
    Returns:
        np.array: filtered data
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def zero_lag_butterworth_filter(data, cutoff_freq=6, sampling_freq=500, order=4, filter_axis=0):
    """
    Apply a zero lag butterworth filter to the data

    Args:
        data (np.array): data to be filtered
        cutoff_freq (float): cutoff frequency for the filter
        sampling_freq (float): sampling frequency of the collected data
        order (int): order of the filter
        filter_axis (int): axis to filter the data

    Returns:
        np.array: filtered data
    """
    b, a = butter(order/2, cutoff_freq, btype='low', analog=False,fs=sampling_freq)
    return filtfilt(b, a, data,axis=filter_axis)

def central_diff(data, dt, filter_axis=0):
    """
    Calculate the joint velocities or accelerations using central difference method
    
    Args:
        data (np.array): data to be differentiated
        dt (float): time step of the data
        filter_axis (int): axis to filter the data
    
    Returns:
        np.array: differentiated data
    """
    return np.gradient(data,dt,axis=filter_axis)

def frequency_domain_diff(data, dt, cutoff_freq, filter_axis=0):
    # not good enough; check the implementation later
    """
    Calculate the joint velocities or accelerations using frequency domain differentiation method
    
    Args:
        data (np.array): data to be differentiated
        dt (float): time step of the data
        filter_axis (int): axis to filter the data
    
    Returns:
        np.array: differentiated data
    """
    # pad the data periodically to avoid edge effects
    data_pad = np.pad(data,((data.shape[0]//2,data.shape[0]//2),(0,0)),'wrap')
    
    # Transform the data to frequency domain (DFT)
    data_fft = fft(data_pad,axis=filter_axis)
    
    # Define the frequency domain filtering window
    freq = fftfreq(2*data.shape[0],dt)
    freq_window = np.zeros((2*data.shape[0],data.shape[1]))
    freq_window[freq<cutoff_freq,:] = 1
    
    # Apply the window to the frequency domain data
    data_fft_filtered = data_fft*freq_window
    
    # Multiply the filtered data by the frequency domain differentiation factor to get the differentiated data
    # first derivative
    diff_factor = 2*np.pi*1j*np.tile(freq,(data.shape[1],1)).T
    diff1_data_fft = data_fft_filtered*diff_factor
    # second derivative
    diff2_data_fft = diff1_data_fft*diff_factor
    
    # Transform the data back to time domain (IDFT)
    diff1_data = np.real(ifft(diff1_data_fft,axis=filter_axis))
    diff2_data = np.real(ifft(diff2_data_fft,axis=filter_axis))
    
    return diff1_data,diff2_data
        



if __name__=="__main__":
    # Load the data
    data_path="sysid_data/kuka_iiwa_14/kuka_iiwa14_data.csv"
    data = pd.read_csv(data_path)
    # Get the time, joint position and torque data
    torque_data = data.iloc[:,8:15].to_numpy()
    pos_data = data.iloc[:,1:8].to_numpy()
    time_data = data.iloc[:,0].to_numpy()
    
    # Apply the butterworth filter to the torque data
    filtered_torque = zero_lag_butterworth_filter(torque_data, cutoff_freq=2, sampling_freq=500, order=12,filter_axis=0)

    # Plot the filtered and unfiltered torque data
    fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs1[i].plot(torque_data[:,i],label="unfiltered")
        axs1[i].plot(filtered_torque[:,i],label="filtered")
        axs1[i].set_ylabel("Torque (Nm)")
        axs1[i].set_title("Joint "+str(i+1))
        axs1[i].legend()
    axs1[-1].set_xlabel("Time step")
    # plt.show()
    
    # Apply the butterworth filter to the position data
    filtered_pos = zero_lag_butterworth_filter(pos_data, cutoff_freq=2, sampling_freq=500, order=12,filter_axis=0)
    
    # Plot the filtered and unfiltered position data
    fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs2[i].plot(pos_data[:,i],label="unfiltered")
        axs2[i].plot(filtered_pos[:,i],label="filtered")
        axs2[i].set_ylabel("Position (rad)")
        axs2[i].set_title("Joint "+str(i+1))
        axs2[i].legend()
    axs2[-1].set_xlabel("Time step")
    # plt.show()
    
    # Calculate the joint velocities and accelerations using central difference method
    vel_data = central_diff(pos_data, dt=0.002,filter_axis=0)
    acc_data = central_diff(vel_data, dt=0.002,filter_axis=0)
    
    # Plot the joint velocities and accelerations
    fig3, axs3 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    for i in range(7):
        axs3[i,0].plot(vel_data[:,i])
        axs3[i,0].set_ylabel("Velocity (rad/s)")
        axs3[i,0].set_title("Joint "+str(i+1))
        axs3[i,1].plot(acc_data[:,i])
        axs3[i,1].set_ylabel("Acceleration (rad/s^2)")
        axs3[i,1].set_title("Joint "+str(i+1))
    axs3[-1,0].set_xlabel("Time step")
    axs3[-1,1].set_xlabel("Time step")
    # plt.show()
    
    # calculate the joint velocities and accelerations using central difference method with filtered data
    vel_data_filtered = central_diff(filtered_pos, dt=0.002,filter_axis=0)
    acc_data_filtered = central_diff(vel_data_filtered, dt=0.002,filter_axis=0)
    
    # Plot the joint velocities and accelerations
    fig4, axs4 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    for i in range(7):
        axs4[i,0].plot(vel_data_filtered[:,i])
        axs4[i,0].set_ylabel("Velocity (rad/s)")
        axs4[i,0].set_title("Joint "+str(i+1))
        axs4[i,1].plot(acc_data_filtered[:,i])
        axs4[i,1].set_ylabel("Acceleration (rad/s^2)")
        axs4[i,1].set_title("Joint "+str(i+1))
    axs4[-1,0].set_xlabel("Time step")
    axs4[-1,1].set_xlabel("Time step")
    plt.show()
    
    # save the filtered data following as: time, joint positions, joint torques, joint velocities, joint accelerations
    header =["time", "mp1", "mp2", "mp3", "mp4", "mp5", "mp6", "mp7", "mt1", "mt2", "mt3", "mt4", "mt5", "mt6", "mt7", "mv1", "mv2", "mv3", "mv4", "mv5", "mv6", "mv7", "ma1", "ma2", "ma3", "ma4", "ma5", "ma6", "ma7"]
    filtered_data = np.concatenate((time_data.reshape(-1,1),filtered_pos,filtered_torque,vel_data_filtered,acc_data_filtered),axis=1)
    pd.DataFrame(filtered_data,columns=header).to_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_filtered.csv",index=False)
    
    # save the unfiltered data following as: time, joint positions, joint torques, joint velocities, joint accelerations
    header =["time", "mp1", "mp2", "mp3", "mp4", "mp5", "mp6", "mp7", "mt1", "mt2", "mt3", "mt4", "mt5", "mt6", "mt7", "mv1", "mv2", "mv3", "mv4", "mv5", "mv6", "mv7", "ma1", "ma2", "ma3", "ma4", "ma5", "ma6", "ma7"]
    unfiltered_data = np.concatenate((time_data.reshape(-1,1),pos_data,torque_data,vel_data,acc_data),axis=1)
    pd.DataFrame(unfiltered_data,columns=header).to_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_unfiltered.csv",index=False)
    """To do: Test freqeuency domain filtering and differentiation method"""
    
    # # calculate joint pos fft
    # pos_fft = fft(pos_data,axis=0)
    # pos_freq = fftfreq(pos_data.shape[0],0.002)
    
    # # plot the fft of the joint positions
    # fig4, axs4 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    # for i in range(7):
    #     axs4[i].plot(pos_freq,np.abs(pos_fft[:,i]))
    #     # axs4[i].plot(pos_freq[:pos_data.shape[0]//2],np.abs(pos_fft[:pos_data.shape[0]//2,i]))
    #     axs4[i].set_ylabel("Amplitude")
    #     axs4[i].set_title("Joint "+str(i+1))
    # axs4[-1].set_xlabel("Frequency (Hz)")
    # plt.show()
    
    # Calculate the joint velocities and accelerations using frequency domain differentiation method
    # vel_data_freq, acc_data_freq = frequency_domain_diff(pos_data, dt=0.002, cutoff_freq=0.001,filter_axis=0)
    
    # Plot the joint velocities and accelerations
    # fig5, axs5 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    # for i in range(7):
    #     axs5[i,0].plot(vel_data_freq[:,i])
    #     axs5[i,0].set_ylabel("Velocity (rad/s)")
    #     axs5[i,0].set_title("Joint "+str(i+1))
    #     axs5[i,1].plot(acc_data_freq[:,i])
    #     axs5[i,1].set_ylabel("Acceleration (rad/s^2)")
    #     axs5[i,1].set_title("Joint "+str(i+1))
    # axs5[-1,0].set_xlabel("Time step")
    # axs5[-1,1].set_xlabel("Time step")
    # plt.show()
    
