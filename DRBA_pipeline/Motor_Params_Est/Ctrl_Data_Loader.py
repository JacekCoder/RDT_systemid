import pandas as pd
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize


class Ctrl_Data_Loader:
    def __init__(self, feedback_file_path, ctrl_file_path,set_index):
        self.feedback_file_path = feedback_file_path
        self.ctrl_file_path = ctrl_file_path
        self.set_index = set_index
        self.load_data()
        self.process_data()

    def load_data(self):
        # Load the ctrl data
        df_ctrl = pd.read_csv(self.ctrl_file_path)
        self.ctrl_signal = df_ctrl['Ctrl'].values
        self.ctrl_time = df_ctrl['Time'].values
        self.ctrl_time = self.ctrl_time - self.ctrl_time[0]
        self.ctrl_time_interval = np.diff(self.ctrl_time)
        self.avg_time_interval = round(np.mean(self.ctrl_time_interval), 4)

        # Load the feedback file
        df = pd.read_csv(self.feedback_file_path)
        self.time_actual = df['Time'].values
        self.speed_actual = df['Speed'].values
        self.position_actual = df['Position'].values
    
    def process_data(self):
        self.get_settings()
        self.ref_traj_generation()
        self.get_start_end_index()
        self.extract_data()
        
    def get_settings(self):
        self.set_timestep = 0.005
        self.rpm2rad = 2 * np.pi / 60
        self.set_sampling_freq = 1 / self.set_timestep
        if self.set_index == 1:
            self.freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
            self.magnitude = 90 * self.rpm2rad
        if self.set_index == 2:
            self.freqs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.5]
            self.magnitude = 120 * self.rpm2rad
        if self.set_index == 3:
            self.freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
            self.magnitude = 60 * self.rpm2rad
        if self.set_index == 4:
            self.freqs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
            self.magnitude = 30 * self.rpm2rad
    
    def ref_traj_generation(self):
        # reference trajectory generation
        self.ref_signal = []
        self.steps_freq = []
        for freq in self.freqs:
            steps_per_freq = int(self.set_sampling_freq / freq)
            self.steps_freq.append(steps_per_freq)
            for i in range(steps_per_freq):
                self.ref_signal.append(self.magnitude * np.sin(2 * np.pi * freq * i * self.set_timestep))
        self.ref_signal = np.array(self.ref_signal)
        self.steps_freq = np.array(self.steps_freq)
        self.ref_time = np.arange(0, len(self.ref_signal) * self.avg_time_interval, self.avg_time_interval)
    
    def get_start_end_index(self):
        # Find the initial time index where the speed is close to 0
        index_1 = np.where(self.position_actual == 1)[0][0]
        index_2 = np.where(self.position_actual == 2)[0][0]
        assert index_1 is not None and index_2 is not None, "Initial time index not found"
        self.index_begin = index_1 - (index_2 - index_1)
        self.index_end = self.index_begin + int(len(self.ref_signal) * self.avg_time_interval / self.set_timestep)
    
    def extract_data(self):
        # Extract the relevant portion of the data
        self.speed_actual_cut = self.speed_actual[self.index_begin:self.index_end]
        self.pos_actual_cut = self.position_actual[self.index_begin:self.index_end] * np.pi / 12 / 22.67
        self.time_cut = self.time_actual[self.index_begin:self.index_end]
        self.time_cut = self.time_cut - self.time_cut[0]
    
    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.ref_time, self.ref_signal, label='Speed Reference', color='red')
        plt.plot(self.time_cut, self.speed_actual_cut, label='Speed Actual', color='blue')
        # plt.plot(self.ctrl_time, self.ctrl_signal, "*", label='Speed Reference', color='red')
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_actual, self.position_actual, "o", label='Position Actual', color='blue')
        plt.show()
    
    def get_weights_freq(self):
        # Calculate the weights for each frequency
        self.weights_freq = np.sum(self.steps_freq)/ np.array(self.steps_freq)
        self.weights_freq = self.weights_freq / np.sum(self.weights_freq)

if __name__ == "__main__":
    feedback_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1.csv"
    ctrl_file_path = "sysid_data/DRBA/motor/sysid/canalystII/set3_has_1_ctrl.csv"
    set_index = 3
    data_loader = Ctrl_Data_Loader(feedback_file_path, ctrl_file_path, set_index)
    data_loader.plot_data()
    print(data_loader.steps_freq)