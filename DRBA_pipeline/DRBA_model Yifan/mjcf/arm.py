import mujoco
import mujoco.viewer
import numpy as np
import argparse
from scipy.optimize import minimize

# pos parameters
parser = argparse.ArgumentParser(description="IK solve for interface to target pos")
parser.add_argument("x", type=float, help="Target x position")
parser.add_argument("y", type=float, help="Target y position")
parser.add_argument("z", type=float, help="Target z position")
parser.add_argument("--xml", type=str, default="DRBA_v1_arm.xml", help="MuJoCo XML model")
parser.add_argument("--save", type=str, default="ik_result.txt", help="File to save joint angles")
args = parser.parse_args()

TARGET_POS = np.array([args.x, args.y, args.z])
XML_PATH = args.xml

# load xml
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# joint selection
JOINT_NAMES = [
    "interface_ty",
    "L_distal", "L_fore", "L_toInterface",
    "R_distal", "R_fore", "R_toInterface"
]

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
dof_indices = [model.jnt_dofadr[jid] for jid in joint_ids]

initial_qpos = data.qpos[dof_indices].copy()
initial_qpos[0] = 0.0  # interface_ty


# call cost function 
def cost_function(q):
    data.qpos[dof_indices] = q
    mujoco.mj_forward(model, data)
    pos = data.body("interface").xpos
    error = pos - TARGET_POS
    cost = np.linalg.norm(error)
    cost += 20.0 * abs(error[1])
    return cost

# solve inverse kinematics
print("🎯 Solving IK for target:", TARGET_POS)
result = minimize(cost_function, initial_qpos, method="L-BFGS-B")

# results
if result.success:
    q_sol = result.x
    data.qpos[dof_indices] = q_sol
    mujoco.mj_forward(model, data)
    final_pos = data.body("interface").xpos
    print("✅ IK solved.")
    print(f"→ interface moved to: {final_pos}")
    print(f"→ error: {np.linalg.norm(final_pos - TARGET_POS):.6f}")

    # save as txt
    with open(args.save, "w") as f:
        f.write("# IK Result - Joint Angles\n")
        for name, val in zip(JOINT_NAMES, q_sol):
            line = f"{name} = {val:.6f}\n"
            f.write(line)
        f.write(f"\n# Final interface pos: {final_pos.tolist()}\n")
        f.write(f"# Target pos: {TARGET_POS.tolist()}\n")
        f.write(f"# Error: {np.linalg.norm(final_pos - TARGET_POS):.6f}\n")
    print(f"💾 Saved joint angles to '{args.save}'")

else:
    print("❌ IK failed:", result.message)

# viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer launched. Close to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
