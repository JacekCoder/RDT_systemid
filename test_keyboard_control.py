import mujoco
import mujoco.viewer
import numpy as np
import threading
from pynput import keyboard

# Dictory to track the up down left right key press or release
held_keys = {
    "up": False,
    "down": False,
    "left": False,
    "right": False
}

def on_press(key):
    if key == keyboard.Key.up:
        held_keys["up"] = True
    if key == keyboard.Key.down:
        held_keys["down"] = True
    if key == keyboard.Key.left:
        held_keys["left"] = True
    if key == keyboard.Key.right:
        held_keys["right"] = True

def on_release(key):
    if key == keyboard.Key.up:
        held_keys["up"] = False
    if key == keyboard.Key.down:
        held_keys["down"] = False
    if key == keyboard.Key.left:
        held_keys["left"] = False
    if key == keyboard.Key.right:
        held_keys["right"] = False
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def minicar_controller(mj_data):
    if held_keys["up"]:
        mj_data.ctrl[0] -=0.001
    if held_keys["down"]:
        mj_data.ctrl[0] +=0.001
    if held_keys["left"]:
        mj_data.ctrl[1] +=0.001
    if held_keys["right"]:
        mj_data.ctrl[1] -=0.001
    if not held_keys["up"] and not held_keys["down"]:
        mj_data.ctrl[0] = 0
    if not held_keys["left"] and not held_keys["right"]:
        mj_data.ctrl[1] = 0
        
    
# Test keyboard control minicar
if __name__=="__main__":
    # mj_model = mujoco.MjModel.from_xml_path("model/Minicar/minicar.xml")
    mj_model = mujoco.MjModel.from_xml_path("model/Minicar/scene_minicar.xml")
    mj_data = mujoco.MjData(mj_model)
    keyboard_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    keyboard_thread.start()
    with mujoco.viewer.launch_passive(mj_model,mj_data) as viewer:      
        while viewer.is_running():
            minicar_controller(mj_data)
            mujoco.mj_step(mj_model,mj_data)
            viewer.sync()
