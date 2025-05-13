import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from ik_solver import solve_ik

# 模型路径
XML_PATH = "DRBA_v1_arm.xml"

# 参数初始值 (3个joint, 每个joint有两个参数)
initial_params = np.array([0.2, 0.2,   # distal_arm: damping, frictionloss
                           0.2, 0.2,   # fore_arm: damping, frictionloss
                           0.3, 0.3])  # ToInterface: damping, frictionloss

# 目标轨迹集合（示例随机生成，可根据需要自行指定）
def generate_target_positions(num_points=10):
    np.random.seed(0)
    return np.random.uniform(low=[0.3,-0.2,0.8], high=[0.5,0.2,1.0], size=(num_points, 3))

target_positions = generate_target_positions()

# 修改模型参数并保存
def set_model_params(model, params):
    classes = ['distal_arm', 'fore_arm', 'ToInterface']
    for i, cls in enumerate(classes):
        damping, frictionloss = params[2*i], params[2*i+1]
        for j in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            joint_cls = model.jnt_user[j]
            if cls in joint_name:
                model.dof_damping[j] = damping
                model.dof_frictionloss[j] = frictionloss

# cost函数：计算目标轨迹与模型轨迹误差
def cost_function(params):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    set_model_params(model, params)
    data = mujoco.MjData(model)

    total_error = 0
    sim_positions = []

    for target_pos in target_positions:
        q_sol, final_pos, error = solve_ik(model, data, target_pos)
        total_error += error
        sim_positions.append(final_pos.copy())

    avg_error = total_error / len(target_positions)

    # 绘制本轮轨迹
    sim_positions = np.array(sim_positions)
    plt.figure(figsize=(8,6))
    plt.scatter(target_positions[:,0], target_positions[:,2], label="Target", c='r')
    plt.scatter(sim_positions[:,0], sim_positions[:,2], label="Simulation", c='b')
    plt.xlabel("X position")
    plt.ylabel("Z position")
    plt.title(f"Iteration Error: {avg_error:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return avg_error

# 执行优化过程
result = minimize(cost_function, initial_params, method="L-BFGS-B", 
                  bounds=[(0,1)]*6, options={'disp': True})

print("Optimized parameters:", result.x)

