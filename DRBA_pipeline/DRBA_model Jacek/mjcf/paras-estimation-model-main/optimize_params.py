import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import mujoco

# six joints label
labels6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
# three sets of initial angles for the six joints
initial_angles_list = [
    [0.0,  0.0,  0.0,  0.0],
    [0.2,  0.2, -0.2, -0.2],
    [-0.2, -0.2,  0.2,  0.2]
]

# read reeal trajectorys from CSV files 
def load_real_trajectories(data_dir="real_data"):
    real_trajs = []
    for idx in range(1, 4):
        path = os.path.join(data_dir, f"real_traj_set{idx}.csv")
        # jump header line
        real_trajs.append(np.loadtxt(path, delimiter=",", skiprows=1))
    return real_trajs

# Simulation function: replay three sets of trajectories based on the parameters to be fitted
# Reuse the same logic from collect_real.py, without visualization, and read the CSV directly

def simulate_drba(params, init_angles, steps):
    model = mujoco.MjModel.from_xml_path("DRBA_v1_arm.xml")
    data = mujoco.MjData(model)

    # set params
    damping = [params[0], params[2], params[4]]
    friction = [params[1], params[3], params[5]]
    for name in ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[jid]
        if name.endswith('distal'):
            model.dof_damping[dof] = damping[0]
            model.dof_frictionloss[dof] = friction[0]
        elif name.endswith('fore'):
            model.dof_damping[dof] = damping[1]
            model.dof_frictionloss[dof] = friction[1]
        else:
            model.dof_damping[dof] = damping[2]
            model.dof_frictionloss[dof] = friction[2]

    # set initial angles
    for angle, name in zip(init_angles, ["L_distal","L_fore","R_distal","R_fore"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_dofadr[jid]] = angle
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    traj = []
    dofs6 = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] 
             for n in ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]]
    for _ in range(steps):
        mujoco.mj_step(model, data)
        traj.append(data.qpos[dofs6].copy())
    return np.array(traj)

# loss function
def loss_fn(params, real_trajs):
    total = 0.0
    steps = real_trajs[0].shape[0]
    for init, real in zip(initial_angles_list, real_trajs):
        sim = simulate_drba(params, init, steps)
        total += np.sum((sim - real)**2)
    return total / len(real_trajs)

# main function: Load, Fit, Output
if __name__=='__main__':
    real_trajs = load_real_trajectories()
    initial_guess = np.array([0.1]*6)
    bounds = [(0.01,1.0)]*6
    res = minimize(loss_fn, initial_guess, args=(real_trajs,), method='L-BFGS-B', bounds=bounds,
                   options={'disp':True, 'maxiter':50})
    print("\nOptimized Parameters:", np.round(res.x,4))
