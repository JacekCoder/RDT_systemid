import numpy as np
import matplotlib.pyplot as plt
from MjSim.mujoco_base import MuJoCoBase
from RobSim.kuka_iiwa_14.kuka_iiwa_14_pos import kuka_iiwa14_pos
from RobSim.kuka_iiwa_14.kuka_iiwa_14_torque import kuka_iiwa14_torque

def torque_main():
    # Load the model
    xml_path = "model/kuka_iiwa_14/scene_torque.xml"
    
    kukaSim = kuka_iiwa14_torque(xml_path, sim_timestep = 0.002)
    # kukaSim.load_exp_data("sysid_data/kuka_iiwa_14/kuka_iiwa14_data.csv")
    kukaSim.load_exp_data("sysid_data/kuka_iiwa_14/kuka_iiwa14_data_filtered.csv")
    kukaSim.simulate()

    qpos_sim = np.array(kukaSim.qpos_list)
    qfrc_actuator_sim = np.array(kukaSim.qfrc_actuator_list)
    qpos_exp = np.array(kukaSim.exp_data.iloc[:,1:8])
    qfrc_actuator_exp = np.array(kukaSim.exp_data.iloc[:,8:15])

    # plot the simulated and experimental joint pos data
    # subplot for each joint
    fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs1[i].plot(qpos_sim[:,i],label="simulated")
        axs1[i].plot(qpos_exp[:,i],label="experimental")
        axs1[i].set_ylabel("Position (rad)")
        axs1[i].set_title("Joint "+str(i+1))
        axs1[i].legend()
    axs1[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig1.savefig("test_figure/kuka_iiwa_14/Real_Sim_pos_comparison_torque_ctrl.png")
    
    # plot the simulated and experimental joint torque data
    # subplot for each joint
    fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs2[i].plot(qfrc_actuator_sim[:,i],label="simulated")
        axs2[i].plot(qfrc_actuator_exp[:,i],label="experimental")
        axs2[i].set_ylabel("Torque (Nm)")
        axs2[i].set_title("Joint "+str(i+1))
        axs2[i].legend()
    axs2[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig2.savefig("test_figure/kuka_iiwa_14/Real_Sim_torque_comparison_torque_ctrl.png")
    
def pos_main():
    xml_path = "model/kuka_iiwa_14/scene_pos.xml"
    
    kukaSim = kuka_iiwa14_pos(xml_path, sim_timestep = 0.002)
    # kukaSim.load_exp_data("sysid_data/kuka_iiwa_14/kuka_iiwa14_data.csv")
    kukaSim.load_exp_data("sysid_data/kuka_iiwa_14/kuka_iiwa14_data_filtered.csv")
    kukaSim.simulate()

    qpos_sim = np.array(kukaSim.qpos_list)
    qfrc_actuator_sim = np.array(kukaSim.qfrc_actuator_list)
    qpos_exp = np.array(kukaSim.exp_data.iloc[:,1:8])
    qfrc_actuator_exp = np.array(kukaSim.exp_data.iloc[:,8:15])

    # plot the simulated and experimental joint pos data
    # subplot for each joint
    fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs1[i].plot(qpos_sim[:,i],label="simulated")
        axs1[i].plot(qpos_exp[:,i],label="experimental")
        axs1[i].set_ylabel("Position (rad)")
        axs1[i].set_title("Joint "+str(i+1))
        axs1[i].legend()
    axs1[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig1.savefig("test_figure/kuka_iiwa_14/Real_Sim_pos_comparison_pos_ctrl.png")
    
    # plot the simulated and experimental joint torque data
    # subplot for each joint
    fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(7):
        axs2[i].plot(qfrc_actuator_sim[:,i],label="simulated")
        axs2[i].plot(qfrc_actuator_exp[:,i],label="experimental")
        axs2[i].set_ylabel("Torque (Nm)")
        axs2[i].set_title("Joint "+str(i+1))
        axs2[i].legend()
    axs2[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig2.savefig("test_figure/kuka_iiwa_14/Real_Sim_torque_comparison_pos_ctrl.png")

if __name__=="__main__":
    # torque_main()
    pos_main()