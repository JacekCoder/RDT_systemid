import pandas as pd
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from Ctrl_Data_Loader import Ctrl_Data_Loader
from Ctrl_Params_Optimizer import Ctrl_Params_Optimizer

def Params_Estimation(feedback_file_path, ctrl_file_path,set_index,motor_xml_path):
    # Load the data
    data_loader = Ctrl_Data_Loader(feedback_file_path, ctrl_file_path, set_index)
    data_loader.get_weights_freq()
    weights_matrix = []
    for i, steps_per_freq in enumerate(data_loader.steps_freq):
        weights_matrix.extend(np.ones(int(steps_per_freq*len(data_loader.time_cut)/len(data_loader.ref_signal)))*data_loader.weights_freq[i])
    weights_matrix = np.array(weights_matrix)
    weights_matrix2 = np.ones(len(data_loader.time_cut))*weights_matrix[0]
    pad_len = len(weights_matrix2) - len(weights_matrix)
    weights_matrix2[pad_len:] = weights_matrix
    
    # Load the optimizer
    ctrl_opt = Ctrl_Params_Optimizer(motor_xml_path, data_loader.ref_signal, data_loader.pos_actual_cut, data_loader.avg_time_interval, data_loader.set_timestep, weights_matrix2)
    ctrl_opt.set_optimizer(method='L-BFGS-B', initial_guess=[12.0, 3.0], bounds=[(0.1, 100), (0.1, 100)])
    ctrl_opt.optimize_params()
    
    # Extract optimized parameters
    return ctrl_opt.result

def Params_Validation(feedback_file_path, ctrl_file_path,set_index,motor_xml_path,kp,kv):
    # Load the data
    data_loader = Ctrl_Data_Loader(feedback_file_path, ctrl_file_path, set_index)
    data_loader.get_weights_freq()
    weights_matrix = []
    for i, steps_per_freq in enumerate(data_loader.steps_freq):
        weights_matrix.extend(np.ones(int(steps_per_freq*len(data_loader.time_cut)/len(data_loader.ref_signal)))*data_loader.weights_freq[i])
    weights_matrix = np.array(weights_matrix)
    weights_matrix2 = np.ones(len(data_loader.time_cut))*weights_matrix[0]
    pad_len = len(weights_matrix2) - len(weights_matrix)
    weights_matrix2[pad_len:] = weights_matrix
    
    # Load the optimizer
    ctrl_opt = Ctrl_Params_Optimizer(motor_xml_path, data_loader.ref_signal, data_loader.pos_actual_cut, data_loader.avg_time_interval, data_loader.set_timestep, weights_matrix2)
    
    # Validate the parameters
    return ctrl_opt.simulate_actuator_val(kp,kv)

if __name__ == "__main__":
    Est_or_Val = False
    
    if Est_or_Val:
        # Example usage of the Params_Estimation function
        feedback_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1.csv"
        ctrl_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1_ctrl.csv"
        set_index = 3
        motor_xml_path = "DRBA_pipeline/DRBA_model/mjcf/DRBA_v1_motor_intvel.xml"
        
        optimized_params = Params_Estimation(feedback_file_path, ctrl_file_path, set_index, motor_xml_path)
        print("Optimized Parameters:", optimized_params)
    
    else:
        # Example usage of the Params_Validation function
        # Load the data
        feedback_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set4_has_1.csv"
        ctrl_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set4_has_1_ctrl.csv"
        set_index = 4
        data_loader = Ctrl_Data_Loader(feedback_file_path, ctrl_file_path, set_index)
        data_loader.get_weights_freq()
        weights_matrix = []
        for i, steps_per_freq in enumerate(data_loader.steps_freq):
            weights_matrix.extend(np.ones(int(steps_per_freq*len(data_loader.time_cut)/len(data_loader.ref_signal)))*data_loader.weights_freq[i])
        weights_matrix = np.array(weights_matrix)
        weights_matrix2 = np.ones(len(data_loader.time_cut))*weights_matrix[0]
        pad_len = len(weights_matrix2) - len(weights_matrix)
        weights_matrix2[pad_len:] = weights_matrix
        # Load the optimizer
        motor_xml_path = "DRBA_pipeline/DRBA_model/mjcf/DRBA_v1_motor_intvel.xml"
        ctrl_opt = Ctrl_Params_Optimizer(motor_xml_path, data_loader.ref_signal, data_loader.pos_actual_cut, data_loader.avg_time_interval, data_loader.set_timestep, weights_matrix2)
        
        # Validate the parameters    
        kp = 100.0
        kv = 10.82
        sim_pos, sim_vel = ctrl_opt.simulate_actuator_val(kp, kv)
        sim_time = np.arange(0, len(sim_pos)) * data_loader.set_timestep
        
        plt.figure(figsize=(10, 5))
        plt.plot(data_loader.ref_time, data_loader.ref_signal, label='Speed Reference', color='red')
        plt.plot(data_loader.time_cut, data_loader.speed_actual_cut, label='Speed Actual', color='blue')
        plt.plot(sim_time,sim_vel, label='Simulated Position', color='green')
        plt.legend()
        # plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(data_loader.time_cut, data_loader.pos_actual_cut, label='Position Actual', color='blue')
        plt.plot(sim_time, sim_pos, label='Simulated Position', color='green')
        plt.legend()
        plt.show()