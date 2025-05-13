"""
=========================
Script description
=========================
This script is used to construct the optimization function with constraints.
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

from RobSim.kuka_iiwa_14.kuka_iiwa_14_pos import kuka_iiwa14_pos
from RobSim.kuka_iiwa_14.kuka_iiwa_14_torque import kuka_iiwa14_torque
from MjSysid import get_inertial_parameters, set_inertial_parameters, get_damping_parameters, get_frictionloss_parameters

# Load sysid data (filtered)
# sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_filtered.csv")
sysid_data = pd.read_csv("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_unfiltered.csv")
real_tau = -sysid_data.iloc[:,8:15].to_numpy()
real_tau = real_tau.reshape((140098,1))
# print(real_tau.shape)

# Load MuJoCo model
xml_path = "model/kuka_iiwa_14/scene_torque.xml"
kukaSim = kuka_iiwa14_torque(xml_path, sim_timestep = 0.002)

# Load Observation matrix
Y_obs = np.load("sysid_pipeline/03_Validate_Observation_Matrix/Y_obs.npy")
# print(Y_obs.shape)

# Create a dictionary to store the parameters to be identified
param_names_inertial = [f"{prefix}{i}" for i in range(1,8) for prefix in ["M","Mrx","Mry","Mrz","Ixx","Ixy","Ixz","Iyy","Iyz","Izz"]]
param_names_damping = [f"D{i}" for i in range(1,8)]
param_names_frictionloss = [f"F{i}" for i in range(1,8)]
param_names = param_names_inertial + param_names_damping + param_names_frictionloss
param_dict = {name:{"init":None,"Low":None,"Up":None} for name in param_names}

# Initial guess of the parameters to be identified
inertial_coeff = np.concatenate([get_inertial_parameters(kukaSim.model, body_id) for body_id in kukaSim.model.jnt_bodyid])
damping_coeff = np.array([get_damping_parameters(kukaSim.model, jnt_id) for jnt_id in range(kukaSim.model.njnt)])
frictionloss = np.array([get_frictionloss_parameters(kukaSim.model, jnt_id) for jnt_id in range(kukaSim.model.njnt)])
x0 = np.concatenate([inertial_coeff,damping_coeff,frictionloss])
param_dict = {name:{"init":x0[i],"Low":None,"Up":None} for i,name in enumerate(param_names)}

# Constraints of parameters to be identified
# Inertial parameters
# Mass
for i in range(7):
    param_dict[f"M{i+1}"]["Low"] = 1
    param_dict[f"M{i+1}"]["Up"] = 6.5
param_dict["M7"]["Up"] = 5

# Inertia matrix
for i in range(7):
    param_dict[f"Ixx{i+1}"]["Low"] = -1 #1e-10
    param_dict[f"Ixx{i+1}"]["Up"] = 1
    param_dict[f"Iyy{i+1}"]["Low"] = -1 #1e-10
    param_dict[f"Iyy{i+1}"]["Up"] = 1
    param_dict[f"Izz{i+1}"]["Low"] = -1 #1e-10
    param_dict[f"Izz{i+1}"]["Up"] = 1
    param_dict[f"Ixy{i+1}"]["Low"] = -1
    param_dict[f"Ixy{i+1}"]["Up"] = 1
    param_dict[f"Ixz{i+1}"]["Low"] = -1
    param_dict[f"Ixz{i+1}"]["Up"] = 1
    param_dict[f"Iyz{i+1}"]["Low"] = -1
    param_dict[f"Iyz{i+1}"]["Up"] = 1
# Center of mass
for i in range(7):
    param_dict[f"Mrx{i+1}"]["Low"] = -0.05*param_dict[f"M{i+1}"]["Up"]
    param_dict[f"Mrx{i+1}"]["Up"] = 0.05 * param_dict[f"M{i+1}"]["Up"]
    param_dict[f"Mry{i+1}"]["Low"] = -0.05 * param_dict[f"M{i+1}"]["Up"]
    param_dict[f"Mry{i+1}"]["Up"] = 0.05 * param_dict[f"M{i+1}"]["Up"]
    param_dict[f"Mrz{i+1}"]["Low"] = -0.05 * param_dict[f"M{i+1}"]["Up"]
    param_dict[f"Mrz{i+1}"]["Up"] = 0.05 * param_dict[f"M{i+1}"]["Up"]
param_dict["Mrz1"]["Up"] = 0.4 * param_dict["M1"]["Up"]
param_dict["Mry2"]["Up"] = 0.3 * param_dict["M2"]["Up"]
param_dict["Mrz3"]["Up"] = 0.3 * param_dict["M3"]["Up"]
param_dict["Mry4"]["Low"] = -0.3 * param_dict["M4"]["Up"]
param_dict["Mrz5"]["Low"] = -0.3 * param_dict["M5"]["Up"]
param_dict["Mry6"]["Up"] = 0.15 * param_dict["M6"]["Up"]
param_dict["Mrz7"]["Low"] = 0.001 * param_dict["M7"]["Up"]
param_dict["Mrz7"]["Up"] = 0.4 * param_dict["M7"]["Up"]
# Damping parameters
for i in range(7):
    param_dict[f"D{i+1}"]["Low"] = 0
    param_dict[f"D{i+1}"]["Up"] = 1
# Frictionloss parameters
for i in range(7):
    param_dict[f"F{i+1}"]["Low"] = 0
    param_dict[f"F{i+1}"]["Up"] = 1

# Check all initial within low and up bounds
for name in param_names:
    if param_dict[name]["init"] < param_dict[name]["Low"] or param_dict[name]["init"] > param_dict[name]["Up"]:
        # print(f"Initial value of {name} is out of bounds")
        # print(f"Initial value: {param_dict[name]['init']}")
        # print(f"Lower bound: {param_dict[name]['Low']}")
        # print(f"Upper bound: {param_dict[name]['Up']}")
        
        # Adjust the initial value to the nearest bound
        if param_dict[name]["init"] < param_dict[name]["Low"]:
            param_dict[name]["init"] = param_dict[name]["Low"]
        else:
            param_dict[name]["init"] = param_dict[name]["Up"]

# Residual function
def res_tau(x):
    return real_tau - Y_obs@x

# Check initia residual
x0 = np.array([param_dict[name]["init"] for name in param_names])
xlow = np.array([param_dict[name]["Low"] for name in param_names])
xup = np.array([param_dict[name]["Up"] for name in param_names])
bounds =[xlow,xup]

# Minimize the residual function
x_opt, rb_trace = minimize.least_squares(x0, res_tau,bounds=bounds,tol=1e-12)
# x_opt,rb_trace = minimize.least_squares(x0, res_tau,tol=1e-7)


# Check x_opt within low and up bounds
for i,name in enumerate(param_names):
    if x_opt[i] < param_dict[name]["Low"] or x_opt[i] > param_dict[name]["Up"]:
        print(f"Optimized value of {name} is out of bounds")
        print(f"Optimized value: {x_opt[i]}")
        print(f"Lower bound: {param_dict[name]['Low']}")
        print(f"Upper bound: {param_dict[name]['Up']}")

# Save x_opt, rb_trace
np.save("sysid_pipeline/04_Construct_Optimization_Function/x_opt.npy",x_opt)

tau_obs = Y_obs@x_opt
tau_obs = tau_obs.reshape(20014,7)
real_tau = real_tau.reshape(20014,7)
# print(tau_obs.shape)

# plot observed torque and real torque
fig, axs = plt.subplots(7,1,figsize=(10,15),sharex=True)
for i in range(7):
    axs[i].plot(tau_obs[:,i],label="observed")
    axs[i].plot(real_tau[:,i],label="real")
    axs[i].set_ylabel("Torque (Nm)")
    axs[i].set_title("Joint "+str(i+1))
    axs[i].legend()
axs[-1].set_xlabel("Time step")
plt.show()

print(x_opt)
# print(rb_trace)