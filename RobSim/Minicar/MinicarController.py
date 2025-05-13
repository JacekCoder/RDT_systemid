"""
This file contains the MinicaController class, which is responsible for controlling the Minicar with keyboard.
"""
import mujoco
import numpy as np
from pynput import keyboard
import threading

class cMinicarController:
    def __init__(self):
        self.held_keys = {
            "up": False,
            "down": False,
            "left": False,
            "right": False
        }
        self.ctrl_forward = 0.0
        self.ctrl_turn = 0.0
        self.keyboard_thread = threading.Thread(target=self.start_keyboard_listener,daemon=True) # Daemon thread which will exit when the main python program exits
        self.keyboard_thread.start()
    
    def on_press(self,key):
        if key == keyboard.Key.up:
            self.held_keys["up"] = True
        if key == keyboard.Key.down:
            self.held_keys["down"] = True
        if key == keyboard.Key.left:
            self.held_keys["left"] = True
        if key == keyboard.Key.right:
            self.held_keys["right"] = True
    def on_release(self,key):
        if key == keyboard.Key.up:
            self.held_keys["up"] = False
        if key == keyboard.Key.down:
            self.held_keys["down"] = False
        if key == keyboard.Key.left:
            self.held_keys["left"] = False
        if key == keyboard.Key.right:
            self.held_keys["right"] = False
        if key == keyboard.Key.esc:
            # Stop listener
            return False
    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def keyboard_controller(self):
        if self.held_keys["up"]:
            self.ctrl_forward -=0.001
        if self.held_keys["down"]:
            self.ctrl_forward +=0.001
        if self.held_keys["left"]:
            self.ctrl_turn +=0.001
        if self.held_keys["right"]:
            self.ctrl_turn -=0.001
        if not self.held_keys["up"] and not self.held_keys["down"]:
            self.ctrl_forward = 0
        if not self.held_keys["left"] and not self.held_keys["right"]:
            self.ctrl_turn = 0
                
