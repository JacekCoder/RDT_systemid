import mujoco
import mujoco.viewer
import os
import numpy as np
from scipy.optimize import minimize 优化参数用的数值优化库
import matplotlib.pyplot as plt

XML_PATH = "DRBA_v1_arm.xml" 
initial_guess = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  [阻尼1, 摩擦1, 阻尼2, 摩擦2, 阻尼3, 摩擦3]

# initial angles for the three groups
initial_angles_list = [
    [0.0,  0.0,  0.0,  0.0],
    [0.2,  0.2, -0.2, -0.2],
    [-0.2, -0.2,  0.2,  0.2]
]

# 模拟函数：根据 params 设置摩擦/阻尼，设置初始角度后释放
def simulate_drba(params, init_angles, steps, visualize=False, real=False):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # set damping and friction
    if not real:
        # params: [d1,f1, d2,f2, d3,f3]
        damping = [params[0], params[2], params[4]]
        friction = [params[1], params[3], params[5]]
        names6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
        for i,name in enumerate(names6):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = model.jnt_dofadr[j]
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
    names4 = ["L_distal","L_fore","R_distal","R_fore"]
    for angle, name in zip(init_angles, names4):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_dofadr[j]] = angle
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # visualize
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    traj = []
    names6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
    dofs6 = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names6]
    #调用 mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) 获取该关节在模型中的 id。
    #使用模型的属性 model.jnt_dofadr[joint_id] 获取该关节对应的自由度（DoF）在数据数组中的位置索引。
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if visualize:
            viewer.sync()
        traj.append(data.qpos[dofs6].copy()) #data.qpos[dofs6]：取出上述六个自由度对应的当前仿真位置（关节角度）。

    #.copy()：防止后续仿真过程中数据被修改，确保存储的是独立的快照。

    if viewer:
        viewer.close()
    return np.array(traj) #NumPy数组 返回

# calculate the real trajectory real=True
def collect_real_trajectories():
    steps = 500
    real_trajs = []
    for init in initial_angles_list:
        real_trajs.append(simulate_drba(None, init, steps, visualize=True, real=True))
    return real_trajs

# Loss function：fit all three groups
def loss_fn(params, real_trajs): 
    total = 0
    steps = real_trajs[0].shape[0]
    for init, real in zip(initial_angles_list, real_trajs):
        sim = simulate_drba(params, init, steps, visualize=False, real=False)
        total += np.sum((sim - real)**2)
    return total / len(initial_angles_list)

# main function
def optimize_drba():
    # 1) data collection
    real_trajs = collect_real_trajectories()

    # 2) paras id
    bounds = [(0,1)]*6
    res = minimize(loss_fn, initial_guess, args=(real_trajs,), method='L-BFGS-B', bounds=bounds,
                   options={'disp':True, 'maxiter':50})
    print("Optimized:", np.round(res.x,4))

    # 3) plot and compare
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    steps = real_trajs[0].shape[0]
    labels6 = ["L_distal","L_fore","L_toInterface","R_distal","R_fore","R_toInterface"]
    for i, (init, real) in enumerate(zip(initial_angles_list, real_trajs)):
        sim = simulate_drba(res.x, init, steps, visualize=False, real=False)
        ax = axes[i]
        for j in range(6):
            ax.plot(real[:,j], label=f"Real {labels6[j]}")
            ax.plot(sim[:,j],'--',label=f"Sim {labels6[j]}")
        ax.set_title(f"Initial Set {i+1}")
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (rad)')
        ax.grid(True)
    # legend
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=6)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

if __name__=='__main__':
    optimize_drba()
