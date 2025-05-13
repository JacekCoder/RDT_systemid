try:
    from .kuka_iiwa_14 import kuka_iiwa14
except:
    from RDT_systemid_control_toolbox.RobSim.kuka_iiwa_14.kuka_iiwa_14 import kuka_iiwa14

class kuka_iiwa14_torque(kuka_iiwa14):
    def __init__(self, 
                 xml_path, 
                 sim_timestep, 
                 sim_timesleep = None):
        super().__init__(xml_path, sim_timestep,sim_timesleep)
    
    def sysid_actuation(self):
        """
        Actuate the robot with the sysid joint torque data
        """
        # Actuate the robot with the sysid data (mp1-mp7)
        for i in range(self._data.ctrl.shape[0]):
            torque_id= "mt"+str(i+1)
            self._data.ctrl[i] = self._exp_data[torque_id][self._timeframe]
