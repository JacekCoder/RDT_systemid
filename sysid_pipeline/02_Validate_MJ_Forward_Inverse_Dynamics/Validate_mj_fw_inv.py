"""
=========================
Script description
=========================
This script is used to validate the mj_forward and mj_inverse functions in mujoco.py.
After validation, the robot motion can be reproduced in MuJoCo with same qpos, qvel, qacc collected from sysid data.
This step is necessary tp make sure the collected data are consistent without exception values.
Make sure the motion can be reproduced correctly before proceeding to the next step.
Author @ rris-Wyf
"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_path)

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.minimize as minimize
from MjSim.mujoco_base import MuJoCoBase
from RobSim.kuka_iiwa_14.kuka_iiwa_14_pos import kuka_iiwa14_pos
from RobSim.kuka_iiwa_14.kuka_iiwa_14_torque import kuka_iiwa14_torque

if __name__ == "__main__":
    #Load model: use torque control!!!#
    xml_path = "model/kuka_iiwa_14/scene_torque.xml"

    kukaSim = kuka_iiwa14_torque(xml_path, sim_timestep = 0.002)

    # check model options to make sure the integrator to be implicit or implicitfast and constraint solver being disabled
    # （including contact and equality）
    print(kukaSim.model.opt)

    # load sysid data (filtered)
    # sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_filtered.csv")
    sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_unfiltered.csv")
    time = sysid_data.iloc[:,0].to_numpy()
    mp = sysid_data.iloc[:,1:8].to_numpy()
    mv = sysid_data.iloc[:,15:22].to_numpy()
    ma = sysid_data.iloc[:,22:].to_numpy()

    # Validate mj_forward, mj_step and mj_inverse
    # Step 1:
    # Given, qpos, qvel, qacc
    # Calculate qfrc_inverse using mujoco inverse dynamics
    # qfrc_inverse = mj_inverse(qpos,qvel,qacc)
    # Step 2:
    # Given qpos, qvel, tau
    # Step the simulation
    # mj_step(qpos,qvel,qfrc_applied = qfrc_inverse)
    # Validate if qpos_new = qpos, qvel_new = qvel, qacc_new = qacc

    # Initialize datacopy for inverse dynamics
    datacopy_inv = copy.deepcopy(kukaSim.data)
    qacc_list = []
    qpos_list = []
    qvel_list = []
    tau_list = []
    qacc_new_list = []
    qpos_new_list = []
    qvel_new_list = []

    # Inverse dynamics
    for i in range(time.shape[0]):
        qpos = mp[i]
        qvel = mv[i]
        qacc = ma[i]
        
        qacc_list.append(qacc)
        qpos_list.append(qpos)
        qvel_list.append(qvel)
        
        datacopy_inv.qpos = qpos
        datacopy_inv.qvel = qvel
        datacopy_inv.qfrc_applied = np.zeros(shape=(kukaSim.model.nv,))
        datacopy_inv.ctrl = np.zeros(shape=(kukaSim.model.nu,))
        mj.mj_forward(kukaSim.model, datacopy_inv)
        datacopy_inv.qacc = qacc
        mj.mj_inverse(kukaSim.model, datacopy_inv)
        tau_list.append(datacopy_inv.qfrc_inverse.copy())

    tau_list = np.vstack(tau_list)
    # save inverse dynamics torques
    header = ["mt1_inv","mt2_inv","mt3_inv","mt4_inv","mt5_inv","mt6_inv","mt7_inv"]
    pd.DataFrame(tau_list,columns=header).to_csv("sysid_pipeline/02_Validate_MJ_Forward_Inverse_Dynamics/tau_inv.csv",index=False)

    # Forward dynamics
    tau_list2 = []
    kukaSim.reset()
    for i in range(time.shape[0]):
        kukaSim.data.qpos = mp[i]
        kukaSim.data.qvel = mv[i]
        kukaSim.data.qfrc_applied[:] = tau_list[i]
        kukaSim.step()
        tau_list2.append(kukaSim.data.qfrc_applied.copy())
        qacc_new_list.append(kukaSim.data.qacc.copy())
        qpos_new_list.append(kukaSim.data.qpos.copy())
        qvel_new_list.append(kukaSim.data.qvel.copy())

    tau_list2 = np.vstack(tau_list2)

    #plot tau difference
    fig5, axs5 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs5[i].plot(tau_list[:,i],label="tau_inv")
        axs5[i].plot(tau_list2[:,i],label="tau_fwd")
        axs5[i].set_ylabel("Torque (Nm)")
        axs5[i].set_title("Joint "+str(i+1))
    axs5[-1].set_xlabel("Time step")
    plt.legend()
    # plt.show()

    # plot qacc and qacc_new difference
    qacc_arr = np.array(qacc_list)
    qacc_new_arr = np.array(qacc_new_list)
    fig1, axs1 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    for i in range(7):
        axs1[i,0].plot(qacc_arr[:,i],label="qacc")
        axs1[i,0].plot(qacc_new_arr[:,i],label="qacc_new")
        axs1[i,0].set_ylabel("Acceleration (rad/s^2)")
        axs1[i,0].set_title("Joint "+str(i+1))
        axs1[i,0].legend()
        axs1[i,1].plot(qacc_arr[:,i]-qacc_new_arr[:,i],label="qacc-qacc_new")
        axs1[i,1].set_ylabel("Acceleration (rad/s^2)")
        axs1[i,1].set_title("Joint "+str(i+1))
        axs1[i,1].legend()
    axs1[-1,0].set_xlabel("Time step")
    axs1[-1,1].set_xlabel("Time step")
    # plt.show()

    # plot qpos and qpos_new difference
    qpos_arr = np.array(qpos_list)
    qpos_new_arr = np.array(qpos_new_list)
    fig2, axs2 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    for i in range(7):
        axs2[i,0].plot(qpos_arr[:,i],label="qpos")
        axs2[i,0].plot(qpos_new_arr[:,i],label="qpos_new")
        axs2[i,0].set_ylabel("Position (rad)")
        axs2[i,0].set_title("Joint "+str(i+1))
        axs2[i,0].legend()
        axs2[i,1].plot(qpos_arr[:,i]-qpos_new_arr[:,i],label="qpos-qpos_new")
        axs2[i,1].set_ylabel("Position (rad)")
        axs2[i,1].set_title("Joint "+str(i+1))
        axs2[i,1].legend()
    axs2[-1,0].set_xlabel("Time step")
    axs2[-1,1].set_xlabel("Time step")
    # plt.show()

    # plot qvel and qvel_new difference
    qvel_arr = np.array(qvel_list)
    qvel_new_arr = np.array(qvel_new_list)
    fig3, axs3 = plt.subplots(7,2,figsize=(10,15),sharex=True)
    for i in range(7):
        axs3[i,0].plot(qvel_arr[:,i],label="qvel")
        axs3[i,0].plot(qvel_new_arr[:,i],label="qvel_new")
        axs3[i,0].set_ylabel("Velocity (rad/s)")
        axs3[i,0].set_title("Joint "+str(i+1))
        axs3[i,0].legend()
        axs3[i,1].plot(qvel_arr[:,i]-qvel_new_arr[:,i],label="qvel-qvel_new")
        axs3[i,1].set_ylabel("Velocity (rad/s)")
        axs3[i,1].set_title("Joint "+str(i+1))
        axs3[i,1].legend()
    axs3[-1,0].set_xlabel("Time step")
    axs3[-1,1].set_xlabel("Time step")
    plt.show()
    
    # Exam the inverse and forward dynamics difference
    assert np.allclose(tau_list,tau_list2,atol=1e-6), f"Norm difference: {np.linalg.norm(tau_inv_list-tau_obs_list)}"
    assert np.allclose(qpos_arr,qpos_new_arr,atol=0.1), f"Norm difference: {np.linalg.norm(qpos_arr-qpos_new_arr)}"
    assert np.allclose(qvel_arr,qvel_new_arr,atol=0.1), f"Norm difference: {np.linalg.norm(qvel_arr-qvel_new_arr)}"
    assert np.allclose(qacc_arr,qacc_new_arr,atol=1e-6), f"Norm difference: {np.linalg.norm(qacc_arr-qacc_new_arr)}"
    print(f"The forward and inverse dynamics are validated consistent")