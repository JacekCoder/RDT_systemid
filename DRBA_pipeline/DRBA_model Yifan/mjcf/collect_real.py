import mujoco
import mujoco.viewer
import numpy as np
import os
import matplotlib.pyplot as plt

XML_PATH = "DRBA_v1_arm.xml"
# three sets of initial angles for the six joints 
initial_angles_list = [
    [0.0,  0.0,  0.0,  0.0],
    [0.2,  0.2, -0.2, -0.2],
    [-0.2, -0.2,  0.2,  0.2]
]
STEPS = 500
OUTPUT_DIR = "real_data"

# make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# simulation function: release the model with the given initial angles and record the joint trajectories
def simulate_real(init_angles, steps=STEPS, visualize=False):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # set initial angles
    names4 = ["L_distal","L_fore","R_distal","R_fore"]
    for angle, name in zip(init_angles, names4):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_dofadr[jid]] = angle
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data) if visualize else None
    traj = []
    names6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
    dofs6 = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names6]

    for _ in range(steps):
        mujoco.mj_step(model, data)
        if visualize:
            viewer.sync()
        traj.append(data.qpos[dofs6].copy())

    if viewer:
        viewer.close()
    return np.array(traj)

# main process: generate three sets of real trajectorys and save them as CSV files, and plot them
if __name__ == '__main__':
    # six joints label
    names6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
    real_trajs = []

    # 1) collect and save real trajectories
    for idx, init in enumerate(initial_angles_list, start=1):
        traj = simulate_real(init, visualize=True)
        real_trajs.append(traj)
        filename = os.path.join(OUTPUT_DIR, f"real_traj_set{idx}.csv")
        np.savetxt(filename, traj, delimiter=",", header=','.join(names6), comments='')
        print(f"Saved real trajectory {idx} to {filename}")

    # 2) plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, (traj, init) in enumerate(zip(real_trajs, initial_angles_list), start=1):
        ax = axes[idx-1]
        for j, name in enumerate(names6):
            ax.plot(traj[:, j], label=name)
        ax.set_title(f"Initial Angles Set {idx}: {init}")
        ax.set_xlabel("Step")
        if idx == 1:
            ax.set_ylabel("Joint Angle (rad)")
        ax.grid(True)

    # Place the legend uniformly outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
