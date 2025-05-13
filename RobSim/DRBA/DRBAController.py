"""
This file contains the DRBAController class, which is responsible for controlling the DRBA system.
"""
import mujoco
import numpy as np
class cDRBAController:
    def __init__(self, mjModel,mjData):
        self.mjModel = mjModel
        self.mjData = mjData
        self.sample_time = 0.001
        self.timesteps = 0
        
        # Robot geometry
        self.wheel_distance = 0.628
        self.wheel_radius = 0.16 # need further check
        self.arm_distance = 0.298 #0.322
        self.distal_arm = 0.250 #0.255
        self.fore_arm = 0.263 #0.2777
        self.init_distal_angle = 3.12179 # arccos(-249.04/250.0) arccos(-249.951/250.0)
        self.init_fore_angle = np.pi - self.init_distal_angle + 0.95729 # arccos(150.543/263.056) arcos(151.451/263.056)
        
        # Robot settings
        # Linear speed
        self.max_speed = 0.85
        self.min_position = 0.22
        self.max_position = 0.46
        self.slope_speed = self.max_speed / (self.max_position - self.min_position)
        self.bias_speed = -self.slope_speed * self.min_position
        # Turning speed
        self.minyaw = np.pi / 90
        self.maxyaw = np.pi / 4.5
        self.max_rotate_speed = 0.4
        self.slope_rotate = self.max_rotate_speed / (self.maxyaw - self.minyaw)
        self.bias_rotate = -self.slope_rotate * self.minyaw
        
        # Controller variables
        # Upper level controller
        self.ref_left_speed = 0.0
        self.ref_right_speed = 0.0
        self.yaw_left_speed = 0.0
        self.yaw_right_speed = 0.0
        # Lower level controller
        self.filtered_left_speed = 0.0
        self.filtered_right_speed = 0.0
        self.integral_eL = 0.0
        self.integral_eR = 0.0
        self.lc_kp = 2.5
        self.lc_ki = 2.0
        self.rc_kd = 2.5
        self.rc_ki = 2.0
        
        self.ctrl_L = 0.0
        self.ctrl_R = 0.0
        # low pass filter
        self.lp_alpha = 0.05
        
        # Human input estimation
        self.D_user = 0.1285120
        self.beta1 = 1
        self.beta2 = 1/(3*self.sample_time)
        self.beta3 = 1
        self.beta4 = 1/(3*self.sample_time)
        self.y_offset = 0.1515
        
        # Data logging
        self.L_ref_speed = []
        self.R_ref_speed = []
        self.L_filtered_speed = []
        self.R_filtered_speed = []
        self.L_raw_speed = []
        self.R_raw_speed = []
        
        self.user_x_log = []
        self.user_y_log = []
        self.user_theta_log = []
        self.user_x_hat_log = []
        self.user_y_hat_log = []
        self.user_theta_hat_log = []
        self.user_dx_hat_log = []
        self.user_dy_hat_log = []
        self.user_dw_hat_log = []
        
        self.L_mini = []
        self.R_mini = []
        
        # init functions
        self.init_mjsim_params()
        
    def init_mjsim_params(self):
        # Get useful joint ID
        self.Left_motor_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "Left_motor")
        self.Right_motor_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "Right_motor")
        self.L_distal_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "L_distal")
        self.L_fore_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "L_fore")
        self.R_distal_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "R_distal")
        self.R_fore_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "R_fore")
        
        self.L_mini_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "joint_left_wheel")
        self.R_mini_JointID = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_JOINT, "joint_right_wheel")
        # Init qpos and qvel data
        self.mQpos = self.mjData.qpos
        self.mQvel = self.mjData.qvel
    
    def update_mjsim_params(self,mjData):
        self.mjData = mjData
        self.mQpos = self.mjData.qpos
        self.mQvel = self.mjData.qvel
        
    
    def caculate_interface_pos(self):
        # Point A and B
        PtA_x = -self.arm_distance/2
        PtA_y = 0
        PtB_x = self.arm_distance/2
        PtB_y = 0
        # Point C and D
        theta1 = self.mQpos[self.L_distal_JointID]+self.init_distal_angle
        theta2 = self.mQpos[self.R_distal_JointID]+self.init_distal_angle
        PtC_x = PtA_x + self.distal_arm * np.cos(theta1)
        PtC_y = PtA_y + self.distal_arm * np.sin(theta1)
        PtD_x = PtB_x - self.distal_arm * np.cos(theta2)
        PtD_y = PtB_y + self.distal_arm * np.sin(theta2)
        # Point E and F
        theta3 = self.mQpos[self.L_fore_JointID]+self.init_fore_angle - (np.pi - theta1)
        theta4 = self.mQpos[self.R_fore_JointID]+self.init_fore_angle - (np.pi - theta2)
        PtM_x = PtC_x + self.fore_arm * np.cos(theta3)
        PtM_y = PtC_y + self.fore_arm * np.sin(theta3)
        PtN_x = PtD_x - self.fore_arm * np.cos(theta4)
        PtN_y = PtD_y + self.fore_arm * np.sin(theta4)
        
        # Calculate the interface position
        self.int_x = (PtM_x + PtN_x) / 2
        self.int_y = (PtM_y + PtN_y) / 2
        self.int_alpha = np.atan2(self.int_y, self.int_x)
        self.int_theta = np.atan2(PtN_y - PtM_y, PtN_x - PtM_x)
        self.int_distance = np.sqrt(self.int_x**2 + self.int_y**2)
        # print(self.int_x,self.int_y,self.int_alpha,self.int_theta,self.int_distance)
        
    def basic_upper_level_controller(self):

        turn_angle = 0.0
        turn_angle1 = 0.0
        
        turn_angle = self.int_alpha - np.pi/2
        if self.int_alpha <= np.pi/2:
            turn_angle1 = np.pi - 2 * self.int_alpha
        else:
            turn_angle1 = np.pi - 2*(np.pi - self.int_alpha)
        
        # Go straight
        if turn_angle > -np.pi/15 and turn_angle < np.pi/15:
            if self.int_distance < self.min_position:
                self.ref_left_speed = 0.0
                self.ref_right_speed = 0.0
            elif self.int_distance < self.max_position:
                self.ref_left_speed = self.slope_speed * self.int_distance + self.bias_speed
                self.ref_right_speed = self.slope_speed * self.int_distance + self.bias_speed
            else:
                self.ref_left_speed = self.max_speed
                self.ref_right_speed = self.max_speed
        # Turn left
        else:
            # arc value
            arc_length = turn_angle1*self.int_distance/(2*np.sin(turn_angle1/2))
            arc_radius = self.int_distance/np.abs(2*np.sin(turn_angle1/2))
            
            if arc_length < self.min_position:
                self.ref_left_speed = 0.0
                self.ref_right_speed = 0.0
            else:
                center_speed = self.slope_speed * arc_length + self.bias_speed
                
                # Do not know why 0.31 but I just follow the original code. Apparently, I should have a more reasonable calculation. Check here futher!!!
                # turn right
                if turn_angle < 0:
                    self.ref_left_speed = center_speed + (center_speed/arc_radius*self.wheel_distance/2)*0.31
                    self.ref_right_speed = center_speed - (center_speed/arc_radius*self.wheel_distance/2)*0.31
                
                # turn left
                else:
                    self.ref_left_speed = center_speed - (center_speed/arc_radius*self.wheel_distance/2)*0.31
                    self.ref_right_speed = center_speed + (center_speed/arc_radius*self.wheel_distance/2)*0.31
        
        if self.int_distance > self.min_position:
            if np.abs(self.int_theta)>=self.minyaw and np.abs(self.int_theta)<=self.maxyaw:
                # turn left
                if self.int_theta > 0:
                    self.yaw_right_speed = self.slope_rotate * np.abs(self.int_theta) + self.bias_rotate
                    self.yaw_left_speed = -self.yaw_right_speed
                # turn right
                else:
                    self.yaw_left_speed = self.slope_rotate * np.abs(self.int_theta) + self.bias_rotate
                    self.yaw_right_speed = -self.yaw_left_speed
            elif np.abs(self.int_theta) > self.maxyaw:
                # turn left
                if self.int_theta > 0:
                    self.yaw_right_speed = self.max_rotate_speed
                    self.yaw_left_speed = -self.yaw_right_speed
                # turn right
                else:
                    self.yaw_left_speed = self.max_rotate_speed
                    self.yaw_right_speed = -self.yaw_left_speed
        
        # combine the speed
        self.ref_left_speed += self.yaw_left_speed*0.7
        self.ref_right_speed += self.yaw_right_speed*0.7
        
        # transform the speed to rotational speed
        self.ref_left_speed = self.ref_left_speed/self.wheel_radius
        self.ref_right_speed = self.ref_right_speed/self.wheel_radius
        
    
    def lower_level_controller(self):
        
        e_L = self.ref_left_speed - self.filtered_left_speed
        e_R = self.ref_right_speed - self.filtered_right_speed
        
        self.integral_eL += e_L * self.sample_time
        self.integral_eR += e_R * self.sample_time
        
        # PI controller
        self.ctrl_L = self.lc_kp * e_L + self.lc_ki * self.integral_eL
        self.ctrl_R = self.rc_kd * e_R + self.rc_ki * self.integral_eR
    
    def human_input_estimation(self):
        """
        Estimate the human input from the interface position and velocity feedback based on the extended state observer (ESO).
        """
        self.user_x = self.int_x - self.D_user*np.sin(self.int_theta)
        self.user_y = self.int_y + self.D_user*np.cos(self.int_theta)+self.y_offset
        self.user_theta = self.int_theta
        
        # test code for non human pos offset
        # self.user_x = self.int_x
        # self.user_y = self.int_y+self.y_offset
        # self.user_theta = self.int_theta
        
        vr = 0.5*(self.filtered_left_speed + self.filtered_right_speed)*self.wheel_radius
        wr = (self.filtered_right_speed - self.filtered_left_speed)*self.wheel_radius/self.wheel_distance
        vy = -vr - self.user_x*wr
        vx = self.user_y*wr
        e1 = self.user_x_hat - self.user_x
        e2 = self.user_y_hat - self.user_y
        
        self.user_dx_hat = self.user_dx_hat - self.beta2*e1
        self.user_x_hat = self.user_x_hat + (vx + self.user_dx_hat)*self.sample_time - self.beta1*e1
        self.user_dy_hat = self.user_dy_hat - self.beta4*e2
        self.user_y_hat = self.user_y_hat + (vy + self.user_dy_hat)*self.sample_time - self.beta3*e2
        
        
    def lp_filter(self,xn,yn):
        yn = (1 - self.lp_alpha) * yn + self.lp_alpha * xn
        return yn
    
    def filter_wheel_vel(self):
        self.filtered_left_speed = self.lp_filter(self.mQvel[self.Left_motor_JointID],self.filtered_left_speed)
        self.filtered_right_speed = self.lp_filter(self.mQvel[self.Right_motor_JointID],self.filtered_right_speed)
    
    
    def init_controller(self):
        self.caculate_interface_pos()
        self.user_y = self.int_y + self.D_user*np.cos(self.int_theta)+self.y_offset
        self.user_x = self.int_x - self.D_user*np.sin(self.int_theta)
        self.user_theta = self.int_theta
        
        # test code for non human pos offset
        # self.user_y = self.int_y + self.y_offset
        # self.user_x = self.int_x
        # self.user_theta = self.int_theta
        
        self.user_y_hat = self.user_y
        self.user_x_hat = self.user_x
        self.user_theta_hat = self.user_theta
        self.user_dx_hat = 0.0
        self.user_dy_hat = 0.0
        self.user_dw_hat = 0.0
        
    def step_controller(self):
        """
        Step DRBA controller with timescale separation. 
        For lower level controller, samping frequency is 1000Hz (time step is 0.001s).
        For upper level controller, sampling frequency is 15Hz (time step is 0.0667s).
        """
        self.filter_wheel_vel()
        self.caculate_interface_pos()
        self.human_input_estimation()
        if self.timesteps % 67 == 0:
            self.basic_upper_level_controller()
        self.lower_level_controller()
        self.timesteps += 1
    
    def log_data(self):
        self.L_ref_speed.append(self.ref_left_speed)
        self.R_ref_speed.append(self.ref_right_speed)
        self.L_filtered_speed.append(self.filtered_left_speed)
        self.R_filtered_speed.append(self.filtered_right_speed)
        self.L_raw_speed.append(self.mQvel[self.Left_motor_JointID])
        self.R_raw_speed.append(self.mQvel[self.Right_motor_JointID])
        self.user_x_log.append(self.user_x)
        self.user_y_log.append(self.user_y)
        self.user_theta_log.append(self.user_theta)
        self.user_x_hat_log.append(self.user_x_hat)
        self.user_y_hat_log.append(self.user_y_hat)
        self.user_dx_hat_log.append(self.user_dx_hat)
        self.user_dy_hat_log.append(self.user_dy_hat)
        self.L_mini.append(self.mQvel[self.L_mini_JointID])
        self.R_mini.append(self.mQvel[self.R_mini_JointID])

                
