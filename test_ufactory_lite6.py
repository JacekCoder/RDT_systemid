import numpy as np
import matplotlib.pyplot as plt
from MjSim.mujoco_base import MuJoCoBase
from RobSim.ufactory_lite6.ufactory_lite6_pos import ufactory_lite6_pos
from RobSim.ufactory_lite6.ufactory_lite6_torque import ufactory_lite6_torque

def torque_main():
    # Load the model
    xml_path = "model/ufactory_lite6/scene_torque.xml"
    
    Lite6Sim = ufactory_lite6_torque(xml_path, sim_timestep = 0.01, sim_timesleep = 0.01)
    Lite6Sim.load_exp_data("sysid_data/ufactory_lite6/joint_pos_and_torques_lite_6_100hz.csv")
    Lite6Sim.set_initial_state()
    Lite6Sim.simulate()

    qpos_sim = np.array(Lite6Sim.qpos_list)
    qfrc_actuator_sim = np.array(Lite6Sim.qfrc_actuator_list)
    qpos_exp = np.array(Lite6Sim.exp_data.iloc[:,1:8])
    qfrc_actuator_exp = np.array(Lite6Sim.exp_data.iloc[:,8:15])

    # plot the simulated and experimental joint pos data
    # subplot for each joint
    fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(6):
        axs1[i].plot(qpos_sim[:,i],label="simulated")
        axs1[i].plot(qpos_exp[:,i],label="experimental")
        axs1[i].set_ylabel("Position (rad)")
        axs1[i].set_title("Joint "+str(i+1))
        axs1[i].legend()
    axs1[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig1.savefig("test_figure/ufactory_lite6/Real_Sim_pos_comparison_torque_ctrl.png")
    
    # plot the simulated and experimental joint torque data
    # subplot for each joint
    fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(6):
        axs2[i].plot(qfrc_actuator_sim[:,i],label="simulated")
        axs2[i].plot(qfrc_actuator_exp[:,i],label="experimental")
        axs2[i].set_ylabel("Torque (Nm)")
        axs2[i].set_title("Joint "+str(i+1))
        axs2[i].legend()
    axs2[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig2.savefig("test_figure/ufactory_lite6/Real_Sim_torque_comparison_torque_ctrl.png")

def pos_main():
    xml_path = "model/ufactory_lite6/scene_pos.xml"
    
    Lite6Sim = ufactory_lite6_pos(xml_path, sim_timestep = 0.01, sim_timesleep = 0.01)
    Lite6Sim.load_exp_data("sysid_data/ufactory_lite6/joint_pos_and_torques_lite_6_100hz.csv")
    Lite6Sim.set_initial_state()
    Lite6Sim.simulate()

    qpos_sim = np.array(Lite6Sim.qpos_list)
    qfrc_actuator_sim = np.array(Lite6Sim.qfrc_actuator_list)
    qpos_exp = np.array(Lite6Sim.exp_data.iloc[:,1:8])
    qfrc_actuator_exp = np.array(Lite6Sim.exp_data.iloc[:,8:15])

    # plot the simulated and experimental joint pos data
    # subplot for each joint
    fig1, axs1 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(6):
        axs1[i].plot(qpos_sim[:,i],label="simulated")
        axs1[i].plot(qpos_exp[:,i],label="experimental")
        axs1[i].set_ylabel("Position (rad)")
        axs1[i].set_title("Joint "+str(i+1))
        axs1[i].legend()
    axs1[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig1.savefig("test_figure/ufactory_lite6/Real_Sim_pos_comparison_pos_ctrl.png")
    
    # plot the simulated and experimental joint torque data
    # subplot for each joint
    fig2, axs2 = plt.subplots(7,1,figsize=(10,15),sharex=True)
    for i in range(6):
        axs2[i].plot(qfrc_actuator_sim[:,i],label="simulated")
        axs2[i].plot(qfrc_actuator_exp[:,i],label="experimental")
        axs2[i].set_ylabel("Torque (Nm)")
        axs2[i].set_title("Joint "+str(i+1))
        axs2[i].legend()
    axs2[-1].set_xlabel("Time step")
    plt.show()
    # save the plot
    fig2.savefig("test_figure/ufactory_lite6/Real_Sim_torque_comparison_pos_ctrl.png")

if __name__=="__main__":
    torque_main()
    # pos_main()