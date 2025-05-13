"""
=========================
Script description
=========================
This script is used to set the optimized parameters for MuJoCo model.
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

from RobSim.kuka_iiwa_14.kuka_iiwa_14_pos import kuka_iiwa14_pos
from RobSim.kuka_iiwa_14.kuka_iiwa_14_torque import kuka_iiwa14_torque
from MjSysid import set_inertial_parameters, set_damping_parameters, set_frictionloss_parameters

opt_param = np.load("sysid_pipeline/04_Construct_Optimization_Function/x_opt.npy")

# Load MuJoCo model
xml_path = "model/kuka_iiwa_14/scene_pos.xml"
kukaSim = kuka_iiwa14_torque(xml_path, sim_timestep = 0.002)

# Set the inertial parameters
for i,body_id in enumerate(kukaSim.model.jnt_bodyid):
    set_inertial_parameters(kukaSim.model, body_id, opt_param[10*i:10*i+10])

# Set the damping parameters
for i in range(kukaSim.model.njnt):
    set_damping_parameters(kukaSim.model, i, opt_param[10*kukaSim.model.njnt+i])

# Set the frictionloss parameters
for i in range(kukaSim.model.njnt):
    set_frictionloss_parameters(kukaSim.model, i, opt_param[11*kukaSim.model.njnt+i])
    
# Save the updated model
mj.mj_saveLastXML("sysid_pipeline/05_Set_Optimized_Parameters/scene_pos_optimized.xml", kukaSim.model)

kukaSim.load_exp_data("sysid_pipeline/01_RealData_Preprocessing/kuka_iiwa14_data_unfiltered.csv")
kukaSim.simulate()
    