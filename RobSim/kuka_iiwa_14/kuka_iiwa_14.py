import mujoco as mj
import numpy as np
import pandas as pd
import time
try:
    from MjSim.mujoco_base import MuJoCoBase
except:
    from RDT_systemid_control_toolbox.MjSim.mujoco_base import MuJoCoBase
import os

class kuka_iiwa14(MuJoCoBase):
    def __init__(self, 
                 xml_path, 
                 sim_timestep, 
                 sim_timesleep = None):
        super().__init__(xml_path, sim_timestep)
        self._timeframe = 0
        self._sim_timesleep = sim_timesleep
        self._qpos_list = []
        self._qfrc_actuator_list = []
        
    def load_exp_data(self,pathTodata):
        """
        Load the experimental data from the csv file
        
        Args:
            pathTodata (str): path to the experimental data
        """
        self._exp_data = pd.read_csv(pathTodata)
        # change mt1-mt7 values to -opposite values
        for i in range(1,8):
            self._exp_data["mt"+str(i)] = -self._exp_data["mt"+str(i)]
    
    def save_data(self):
        self._qpos_list.append(self._data.qpos.copy())
        self._qfrc_actuator_list.append(self._data.qfrc_actuator.copy())
    
    @property
    def qpos_list(self):
        """
        Return the list of joint positions
        """
        return self._qpos_list
    
    @property
    def qfrc_actuator_list(self):
        """
        Return the list of applied joint torques by the actuation
        """
        return self._qfrc_actuator_list
    
    @property
    def exp_data(self):
        """
        Return the experimental data
        """
        return self._exp_data
    
    def sysid_actuation(self):
        """
        Actuate the robot with the sysid joint torque/position data; 
        Depends on the actuation type in the xml file
        """
        raise NotImplementedError("sysid_actuation function is not implemented")
            
    def sysid_animation(self):
        """
        Animate the robot with the sysid joint position data
        """
        # Animate the robot with the sysid data (mp1-mp7)
        for i in range(self._data.qpos.shape[0]):
            tpos_id= "mp"+str(i+1)
            self._data.qpos[i] = self._exp_data[tpos_id][self._timeframe]
            
    def simulate(self):
        while self._viewer.is_running() and self._timeframe < self._exp_data.shape[0]:
            simstart = self._data.time
            self.save_data()
            self.sysid_actuation()
            self.step()
            if self._sim_timesleep is not None:
                time.sleep(self._sim_timesleep)
            self._timeframe += 1
        self.close()
    
    def animation(self):
        while self._viewer.is_running() and self._timeframe < self._exp_data.shape[0]:
            simstart = self._data.time
            self.save_data()
            self.sysid_animation()
            self.forward()
            self._viewer.sync()
            if self._sim_timesleep is not None:
                time.sleep(self._sim_timesleep)
            self._timeframe += 1
            
    def reset(self):
        super().reset()
        self._timeframe = 0