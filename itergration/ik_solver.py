import mujoco
from scipy.optimize import minimize
import numpy as np

JOINT_NAMES = [
    "interface_ty",
    "L_distal", "L_fore", "L_toInterface",
    "R_distal", "R_fore", "R_toInterface"
]

joint_ids_cache = None
dof_indices_cache = None

# IK求解函数
def solve_ik(model, data, TARGET_POS):

    global joint_ids_cache, dof_indices_cache

    if joint_ids_cache is None:
        joint_ids_cache = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        dof_indices_cache = [model.jnt_dofadr[jid] for jid in joint_ids_cache]

    initial_qpos = data.qpos[dof_indices_cache].copy()

    def ik_cost(q):
        data.qpos[dof_indices_cache] = q
        mujoco.mj_forward(model, data)
        pos = data.body("interface").xpos
        error = pos - TARGET_POS
        cost = np.linalg.norm(error) + 20.0 * abs(error[1])
        return cost

    result = minimize(ik_cost, initial_qpos, method="L-BFGS-B")

    if result.success:
        q_sol = result.x
        data.qpos[dof_indices_cache] = q_sol
        mujoco.mj_forward(model, data)
        final_pos = data.body("interface").xpos
        error = np.linalg.norm(final_pos - TARGET_POS)
        return q_sol, final_pos, error
    else:
        raise RuntimeError("IK failed")


