"""
=========================
Script description
=========================
This script is used to create and validate the observation matrix for system identification.
The observation matrix is used to relate the dynamic parameters to the robot motion data.
Validation is to check if the observation matrix can calculate the robot toqrues accurately comparing to mujoco inverse dynamics.
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
from MjSysid import body_regressor,joint_body_regressor,joint_inertial_torque_regressor,joint_damping_torque_regressor,joint_frictionloss_torque_regressor
from MjSysid import get_inertial_parameters, set_inertial_parameters, get_damping_parameters, get_frictionloss_parameters

# Load model: use torque control!!!#
xml_path = "model/kuka_iiwa_14/scene_torque.xml"

kukaSim = kuka_iiwa14_torque(xml_path, sim_timestep = 0.002)

# Check model options to make sure the integrator to be implicit or implicitfast and constraint solver being disabled
# （including contact and equality）
print(kukaSim.model.opt)

# Load sysid data (filtered)
# sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_filtered.csv")
sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_unfiltered.csv")
time = sysid_data.iloc[:,0].to_numpy()
mp = sysid_data.iloc[:,1:8].to_numpy()
mt = sysid_data.iloc[:,8:15].to_numpy()
mv = sysid_data.iloc[:,15:22].to_numpy()
ma = sysid_data.iloc[:,22:29].to_numpy()

"""
parameters to be identified \theta -
parameters to be identified for each link: m,l_cx,l_cy,l_cz,Ixx,Ixy,Ixz,Iyy,Iyz,Izz
parameters to be identified for each joint: damping, (optional) frictionloss, (optional) armature, (optional) stiffness
"""

# Retrieve the inertial parameters of the robot model
inertial_coeff = np.concatenate([get_inertial_parameters(kukaSim.model, body_id) for body_id in kukaSim.model.jnt_bodyid])
damping_coeff = np.array([get_damping_parameters(kukaSim.model, jnt_id) for jnt_id in range(kukaSim.model.njnt)])
frictionloss = np.array([get_frictionloss_parameters(kukaSim.model, jnt_id) for jnt_id in range(kukaSim.model.njnt)])

theta = np.concatenate([inertial_coeff,damping_coeff,frictionloss])

"""
MuJoCo dynamics model: \tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + ...(friction, damping)
can be transformed in a new equaltion represented by observation matrix Y_obs (q,\dot{q},\ddot{q}):
\tau = Y_obs\theta
"""

Y_obs_list = []
tau_inv_list = []
tau_obs_list = []
tau_inertial_list = []
tau_damping_list = []
tau_frictionloss_list = []
for i in range(time.shape[0]):
    qpos = mp[i]
    qvel = mv[i]
    qacc = ma[i]
    
    # Set the robot joint states
    kukaSim.data.qpos[:] = mp[i]
    kukaSim.data.qvel[:] = mv[i]
    kukaSim.data.qacc[:] = ma[i]

    # Calculate the observation matrix
    kukaSim.inverse()
    mj.mj_rnePostConstraint(kukaSim.model, kukaSim.data)
    
    Y_inertial = joint_inertial_torque_regressor(kukaSim.model, kukaSim.data)
    Y_damping = joint_damping_torque_regressor(kukaSim.model, kukaSim.data)
    Y_frictionloss = joint_frictionloss_torque_regressor(kukaSim.model, kukaSim.data)
    Y_obs = np.hstack([Y_inertial,Y_damping,Y_frictionloss])
    
    tau_inertial = Y_inertial@inertial_coeff
    tau_damping = Y_damping@damping_coeff
    tau_frictionloss = Y_frictionloss@frictionloss
    tau_obs = Y_obs@theta
    tau_inv = kukaSim.data.qfrc_inverse.copy()
    
    Y_obs_list.append(Y_obs)
    tau_inv_list.append(tau_inv)
    tau_obs_list.append(tau_obs)
    tau_inertial_list.append(tau_inertial)
    tau_damping_list.append(tau_damping)
    tau_frictionloss_list.append(tau_frictionloss)

# Observation matrix
Y_obs_list = np.vstack(Y_obs_list)

# Inverse dynamics torques
tau_inv_list = np.vstack(tau_inv_list)

# Torques predicted by the observation matrix
tau_obs_list = np.vstack(tau_obs_list)

# Inertial torques
tau_inertial_list = np.vstack(tau_inertial_list)

# Damping torques
tau_damping_list = np.vstack(tau_damping_list)

# Frictionloss torques
tau_frictionloss_list = np.vstack(tau_frictionloss_list)


# Compare the observation matrix predicted torques with the inverse dynamics torques
fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
for i in range(7):
    axs1[i].plot(tau_inv_list[:,i],label="mt"+str(i+1)+"_inv")
    axs1[i].plot(tau_obs_list[:,i],label="mt"+str(i+1)+"_obs")
    axs1[i].legend()

# Compare the predicted frictions with the inverse dynamics frictions
fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
for i in range(7):
    axs2[i].plot(tau_inv_list[:,i]-tau_inertial_list[:,i]-tau_damping_list[:,i],label="inv_fricionloss")
    axs2[i].plot(tau_frictionloss_list[:,i],label="obs_frictionloss")
    axs2[i].legend()
plt.show()
    

# Exame the normalized root mean square error
assert np.allclose(tau_inv_list,tau_obs_list,atol=1e-6),f"Norm difference: {np.linalg.norm(tau_inv_list-tau_obs_list)}"
print(f"The observation matrix is validated successfully with normalized difference:{np.linalg.norm(tau_inv_list-tau_obs_list)}")

# Save the observation matrix
np.save("sysid_pipeline/03_Validate_Observation_Matrix/Y_obs.npy",Y_obs_list)