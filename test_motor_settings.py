import mujoco
import mujoco.viewer
from RobSim.Minicar.MinicarController import cMinicarController

if __name__=="__main__":
    mj_model = mujoco.MjModel.from_xml_path("model/Minicar/scene_minicar.xml")
    mj_data = mujoco.MjData(mj_model)
    minicar_controller = cMinicarController()
    with mujoco.viewer.launch_passive(mj_model,mj_data) as viewer:
        while viewer.is_running():
            minicar_controller.keyboard_controller()
            mj_data.ctrl[0] = minicar_controller.ctrl_forward
            mj_data.ctrl[1] = minicar_controller.ctrl_turn
            mujoco.mj_step(mj_model,mj_data)
            viewer.sync()
    