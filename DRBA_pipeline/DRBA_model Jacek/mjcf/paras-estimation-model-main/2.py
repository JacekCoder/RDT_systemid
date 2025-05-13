import mujoco
import mujoco.viewer
import numpy as np
import argparse
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
XML_DEFAULT    = "DRBA_v1_arm.xml"
JOINTS_SAVE    = ["L_distal","L_fore","R_distal","R_fore"]
# 6 parameters: d1,f1, d2,f2, d3,f3
INITIAL_GUESS  = np.array([0.1,0.1, 0.1,0.1, 0.1,0.1])
BOUNDS         = [(0.0,1.0)]*6
# Three target positions
TARGET_POS_LIST = [
    [0.40,  0.00, 0.00],
    [0.39, 0.10, 0.00],
    [0.43, -0.10, 0.00],
]
# Simulation termination criteria
MAX_STEPS             = 2000
STABLE_THRESHOLD      = 1e-5
REQUIRED_STABLE_STEPS = 50

# ---------------- Utility Functions ----------------

def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    intf_joints = ["interface_tx","interface_ty","interface_tz"]
    intf_jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in intf_joints]
    dof_intf    = [model.jnt_dofadr[j] for j in intf_jids]
    save_jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in JOINTS_SAVE]
    dof_save    = [model.jnt_dofadr[j] for j in save_jids]
    return model, data, dof_intf, dof_save

def solve_ik(model, data, target):
    names8 = [
        "interface_tx", "interface_ty",
        "L_distal","L_fore","L_toInterface",
        "R_distal","R_fore","R_toInterface"
    ]
    jids8  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names8]
    dofs8  = [model.jnt_dofadr[j] for j in jids8]
    q0     = data.qpos[dofs8].copy()
    q0[:2] = target[:2]

    def cost(q):
        data.qpos[dofs8] = q
        mujoco.mj_forward(model, data)
        return np.linalg.norm(data.body("interface").xpos - target)

    res = minimize(cost, q0, method='L-BFGS-B',
                   bounds=[(-0.5,0.5),(-0.5,0.5)] + BOUNDS,
                   options={'maxiter':200,'ftol':1e-6,'disp':False})
    if not res.success:
        print(f"⚠️ IK warning: {res.message}")
        return q0
    return res.x

def safe_initialize_qpos(model, data, dof_intf, ik_q, tolerance=1e-6, max_iter=2000):
    data.qpos[:] = 0
    data.qvel[:] = 0

    joint_names = [
        "interface_tx", "interface_ty", "interface_tz",
        "L_distal", "L_fore", "L_toInterface",
        "R_distal", "R_fore", "R_toInterface"
    ]

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    joint_dofs = [model.jnt_dofadr[jid] for jid in joint_ids]

    if len(ik_q) == 8:
        ik_q_corrected = np.insert(ik_q, 2, 0.0)  # 补上 interface_tz
    else:
        ik_q_corrected = ik_q

    data.qpos[joint_dofs] = ik_q_corrected
    mujoco.mj_forward(model, data)

    for iteration in range(max_iter):
        mujoco.mj_step(model, data)
        constraint_force_norm = np.linalg.norm(data.qfrc_constraint)
        if constraint_force_norm < tolerance:
            break
    else:
        print(f"⚠️ 警告: 未能完全收敛，约束力残差: {constraint_force_norm:.4f}")

    return constraint_force_norm

def simulate_until_stable(model, data, dof_intf, dof_save, ik_q, with_viewer=False):
    safe_initialize_qpos(model, data, dof_intf, ik_q)

    viewer = mujoco.viewer.launch_passive(model, data) if with_viewer else None
    traj_q, traj_v, stable = [], [], 0

    for _ in range(MAX_STEPS):
        mujoco.mj_step(model, data)
        if viewer: viewer.sync()
        q, v = data.qpos[dof_save].copy(), data.qvel[dof_save].copy()
        traj_q.append(q); traj_v.append(v)
        stable = stable+1 if len(traj_q)>1 and np.all(np.abs(q-traj_q[-2])<STABLE_THRESHOLD) else 0
        if stable>=REQUIRED_STABLE_STEPS: break

    if viewer: viewer.close()
    return np.array(traj_q), np.array(traj_v)

def simulate_fixed(model, data, dof_intf, dof_save, ik_q, steps):
    safe_initialize_qpos(model, data, dof_intf, ik_q)
    traj_q, traj_v = [], []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        traj_q.append(data.qpos[dof_save].copy())
        traj_v.append(data.qvel[dof_save].copy())
    return np.array(traj_q), np.array(traj_v)

# Main Flow
if __name__=="__main__":
    model, data, dof_intf, dof_save = load_model(XML_DEFAULT)
    NAMES6 = ["L_distal","L_fore","L_toInterface",
              "R_distal","R_fore","R_toInterface"]

    real_trajs, real_vels, lengths = [], [], []
    for tgt in TARGET_POS_LIST:
        ik_q = solve_ik(model,data,np.array(tgt))
        rq, rv = simulate_until_stable(model,data,dof_intf,dof_save,ik_q,True)
        real_trajs.append(rq); real_vels.append(rv); lengths.append(len(rq))

    def loss_fn(params):
        for i,n in enumerate(NAMES6):
            adr = model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]
            if i%2==0: model.dof_damping[adr]=params[i]
            else:      model.dof_frictionloss[adr]=params[i]
        pos_errs, vel_errs = [],[]
        for idx,tgt in enumerate(TARGET_POS_LIST):
            ik_q = solve_ik(model,data,np.array(tgt))
            sq, sv = simulate_fixed(model,data,dof_intf,dof_save,ik_q,lengths[idx])
            pos_errs.append(np.mean((sq-real_trajs[idx])**2))
            vel_errs.append(np.mean((sv-real_vels[idx])**2))
        return np.mean(pos_errs)+0.1*np.mean(vel_errs)

    res = minimize(loss_fn,INITIAL_GUESS,method='L-BFGS-B',bounds=BOUNDS,options={'maxiter':50,'disp':True})
    print("Optimized Params:",np.round(res.x,4))