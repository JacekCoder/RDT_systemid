import pandas as pd
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from Ctrl_Data_Loader import Ctrl_Data_Loader


class Ctrl_Params_Optimizer:
    def __init__(self, model_path, ref_signal, actual_pos, ctrl_timestep,output_timestep,weight_matrix):
        self.model_path = model_path
        
        self.ref_signal = ref_signal
        self.actual_pos = actual_pos
        
        self.ctrl_timestep = ctrl_timestep
        self.output_timestep = output_timestep
        
        self.weight_matrix = weight_matrix
        
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.n_timestep = int(len(self.ref_signal) * self.ctrl_timestep / self.mj_model.opt.timestep)
        self.n_ctrlstep = self.n_timestep // (int(self.ctrl_timestep / self.mj_model.opt.timestep))
        self.n_outputstep = self.n_timestep // (int(self.output_timestep / self.mj_model.opt.timestep))
        
    
    def update_actuator_gains(self, kp, kv):
        self.mj_model.actuator_gainprm[0][0] = kp
        self.mj_model.actuator_biasprm[0][1] = -kp
        self.mj_model.actuator_biasprm[0][2] = -kv
        mujoco.mj_resetData(self.mj_model, self.mj_data)
    
    def simulate_actuator_est(self, kp, kv):
        self.update_actuator_gains(kp, kv)
        sim_pos = np.zeros(self.n_outputstep)
        
        for i in range(self.n_timestep):
            if i % int(self.ctrl_timestep / self.mj_model.opt.timestep) == 0:
                j = int(i * self.mj_model.opt.timestep / self.ctrl_timestep)
                self.mj_data.ctrl[0] = self.ref_signal[j]
            if i % int(self.output_timestep / self.mj_model.opt.timestep) == 0:
                k = i // int(self.output_timestep / self.mj_model.opt.timestep)
                sim_pos[k] = self.mj_data.qpos[0]
            mujoco.mj_step(self.mj_model, self.mj_data)
        return sim_pos
    
    # Nonlinear least square loss function
    def loss_function(self, params):
        kp, kv = params
        simulated_pos = self.simulate_actuator_est(kp, kv)
        loss = np.sum(self.weight_matrix * (simulated_pos - self.actual_pos)**2)
        print(f"Current loss: {loss}, kp: {kp}, kv: {kv}")
        return loss

    def simulate_actuator_val(self, kp, kv):
        self.update_actuator_gains(kp, kv)
        sim_pos = np.zeros(self.n_outputstep)
        sim_vel = np.zeros(self.n_outputstep)
        
        for i in range(self.n_timestep):
            if i % int(self.ctrl_timestep / self.mj_model.opt.timestep) == 0:
                j = int(i * self.mj_model.opt.timestep / self.ctrl_timestep)
                self.mj_data.ctrl[0] = self.ref_signal[j]
            if i % int(self.output_timestep / self.mj_model.opt.timestep) == 0:
                k = i // int(self.output_timestep / self.mj_model.opt.timestep)
                sim_pos[k] = self.mj_data.qpos[0]
                sim_vel[k] = self.mj_data.qvel[0]
            mujoco.mj_step(self.mj_model, self.mj_data)
        return sim_pos, sim_vel
    
    def set_optimizer(self, method='L-BFGS-B', initial_guess=None, bounds=None):
        if initial_guess is None:
            self.initial_guess = [12.0, 3.0]
        else:
            self.initial_guess = initial_guess
        if bounds is None:
            self.bounds = [(0.1, 100), (0.1, 100)]
        else:
            self.bounds = bounds
        self.method = method
        
    def optimize_params(self):
        result = minimize(self.loss_function, self.initial_guess, bounds=self.bounds, method='L-BFGS-B')
        self.result = result.x


if __name__ == "__main__":
    # Load the data
    feedback_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1.csv"
    ctrl_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1_ctrl.csv"
    set_index = 3
    data_loader = Ctrl_Data_Loader(feedback_file_path, ctrl_file_path, set_index)
    # data_loader.plot_data()
    
    data_loader.get_weights_freq()
    weights_matrix = []
    for i, steps_per_freq in enumerate(data_loader.steps_freq):
        weights_matrix.extend(np.ones(int(steps_per_freq*len(data_loader.time_cut)/len(data_loader.ref_signal)))*data_loader.weights_freq[i])
    weights_matrix = np.array(weights_matrix)
    weights_matrix2 = np.ones(len(data_loader.time_cut))*weights_matrix[0]
    pad_len = len(weights_matrix2) - len(weights_matrix)
    weights_matrix2[pad_len:] = weights_matrix
    
    # Load the optimizer
    motor_xml = "DRBA_pipeline/DRBA_model/mjcf/DRBA_v1_motor_intvel.xml"
    ctrl_opt = Ctrl_Params_Optimizer(motor_xml, data_loader.ref_signal, data_loader.pos_actual_cut, data_loader.avg_time_interval, data_loader.set_timestep, weights_matrix2)
    
    ctrl_opt.set_optimizer(method='L-BFGS-B', initial_guess=[12.0, 3.0], bounds=[(0.1, 1000), (0.1, 100)])
    ctrl_opt.optimize_params()
    print(ctrl_opt.result)
    