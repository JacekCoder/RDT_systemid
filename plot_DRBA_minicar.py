import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lp_filter(xn,yn,lp_alpha=0.05):
    yn = (1 - lp_alpha) * yn + lp_alpha * xn
    return yn


if __name__=="__main__":
    df = pd.read_csv('SimData/DRBA_minicar/DRBA_minicar.csv')
    L1 = df['L_raw_speed']
    R1 = df['R_raw_speed']
    L2 = df['L_filtered_speed']
    R2 = df['R_filtered_speed']
    L3 = df['L_ref_speed']
    R3 = df['R_ref_speed']
    
    int_x = df['int_x']
    int_y = df['int_y']
    int_theta = df['int_theta']
    int_x_hat = df['int_x_hat']
    int_y_hat = df['int_y_hat']
    int_dx_hat = df['int_dx_hat']
    int_dy_hat = df['int_dy_hat']
    
    int_v = np.sqrt(int_dx_hat**2+int_dy_hat**2)
    
    L_mini = df['L_mini']
    R_mini = df['R_mini']
    
    # low pass filter the L_mini and R_mini
    L_mini_filtered = np.zeros_like(L_mini)
    R_mini_filtered = np.zeros_like(R_mini)
    L_mini_filtered[0] = L_mini[0]
    R_mini_filtered[0] = R_mini[0]
    for i in range(1,len(L_mini)):
        L_mini_filtered[i] = lp_filter(L_mini[i],L_mini_filtered[i-1])
        R_mini_filtered[i] = lp_filter(R_mini[i],R_mini_filtered[i-1])
    
    # vr_mini = -(L_mini_filtered+R_mini_filtered)/2*0.05
    vr_mini = -(L_mini+R_mini)/2*0.05
    
    
    # plot the speed data
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(L1,label='L_raw_speed')
    plt.plot(L2,label='L_filtered_speed')
    plt.plot(L3,label='L_ref_speed')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(R1,label='R_raw_speed')
    plt.plot(R2,label='R_filtered_speed')
    plt.plot(R3,label='R_ref_speed')
    plt.legend()
    # plt.show()
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(int_x,label='int_x')
    plt.plot(int_x_hat,label='int_x_hat')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(int_y,label='int_y')
    plt.plot(int_y_hat,label='int_y_hat')
    plt.legend()
    
    plt.figure()
    
    plt.plot(int_v,label='estimated velocity')
    plt.plot(vr_mini,label='measured velocity')
    # plt.plot(int_dx_hat,label='int_dx_hat')
    # plt.plot(int_dy_hat,label='int_dy_hat')
    plt.xlabel('timestep')
    plt.ylabel('velocity (m/s)')
    plt.legend()
    
    plt.figure()
    plt.plot(int_theta,label='int_theta')
    plt.show()