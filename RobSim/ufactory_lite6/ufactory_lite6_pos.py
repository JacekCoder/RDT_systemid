try:
    from .ufactory_lite6 import ufactory_lite6
except:
    from RDT_systemid_control_toolbox.RobSim.ufactory_lite6.ufactory_lite6 import ufactory_lite6

class ufactory_lite6_pos(ufactory_lite6):
    def __init__(self, 
                 xml_path, 
                 sim_timestep, 
                 sim_timesleep = None):
        super().__init__(xml_path, sim_timestep,sim_timesleep)
    
    def sysid_actuation(self):
        """
        Actuate the robot with the sysid joint position data
        """
        # Actuate the robot with the sysid data (mp1-mp7)
        for i in range(self._data.ctrl.shape[0]):
            torque_id= "mp"+str(i+1)
            self._data.ctrl[i] = self._exp_data[torque_id][self._timeframe]
