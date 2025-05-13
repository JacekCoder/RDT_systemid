import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
from pynput import keyboard
from RobSim.DRBA.DRBAController import cDRBAController
from RobSim.Minicar.MinicarController import cMinicarController

if __name__=="__main__":
    mj_model = mujoco.MjModel.from_xml_path("model/DRBA_minicar2.xml")
    mj_data = mujoco.MjData(mj_model)
    drba_controller = cDRBAController(mj_model,mj_data)
    drba_controller.init_mjsim_params()
    drba_controller.init_controller()
    minicar_controller = cMinicarController()
    
    with mujoco.viewer.launch_passive(mj_model,mj_data) as viewer:
        start_time = mj_data.time
        while viewer.is_running():
            minicar_controller.keyboard_controller()
            drba_controller.update_mjsim_params(mj_data)
            drba_controller.step_controller()
            drba_controller.log_data()
            mj_data.ctrl[0] = minicar_controller.ctrl_forward
            mj_data.ctrl[1] = minicar_controller.ctrl_turn
            mj_data.ctrl[2] = drba_controller.ctrl_L
            mj_data.ctrl[3] = drba_controller.ctrl_R
            mujoco.mj_step(mj_model,mj_data)
            if mj_data.time - start_time > 10:
                L1 = np.array(drba_controller.L_raw_speed)
                R1 = np.array(drba_controller.R_raw_speed)
                L2 = np.array(drba_controller.L_filtered_speed)
                R2 = np.array(drba_controller.R_filtered_speed)
                L3 = np.array(drba_controller.L_ref_speed)
                R3 = np.array(drba_controller.R_ref_speed)
                
                int_x = np.array(drba_controller.user_x_log)
                int_y = np.array(drba_controller.user_y_log)
                int_theta = np.array(drba_controller.user_theta_log)
                int_x_hat = np.array(drba_controller.user_x_hat_log)
                int_y_hat = np.array(drba_controller.user_y_hat_log)
                int_dx_hat = np.array(drba_controller.user_dx_hat_log)
                int_dy_hat = np.array(drba_controller.user_dy_hat_log)
                
                L_mini = np.array(drba_controller.L_mini)
                R_mini = np.array(drba_controller.R_mini)
                
                df = pd.DataFrame({'L_raw_speed':L1,'R_raw_speed':R1,'L_filtered_speed':L2,'R_filtered_speed':R2,'L_ref_speed':L3,'R_ref_speed':R3,'int_x':int_x,'int_y':int_y,'int_theta':int_theta,'int_x_hat':int_x_hat,'int_y_hat':int_y_hat,'int_dx_hat':int_dx_hat,'int_dy_hat':int_dy_hat,'L_mini':L_mini,'R_mini':R_mini})
                
                df.to_csv('SimData/DRBA_minicar/DRBA_minicar.csv',index=False)
                break
            viewer.sync()