"""
=========================
Script description
=========================
This script is used to generate the mjcf of the omniwheels. The omniwheel is composed of a plate and a set of rollers.
The plate is a cylinder with a certain radius and height. The rollers are spheres with a certain radius. The rollers are placed on the plate in a circular pattern.
The omniwheel configuration is specified in the config file. The script reads the configuration and generates the mjcf file.
Author @ rris_Wyf and @ JaceKCoder
"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_path)

import numpy as np
import argparse
from utils.config_loader import ConfigLoader
import xml.etree.ElementTree as ET
from xml.dom import minidom


loader = ConfigLoader("DRBA_pipeline/Omniwheel_Generation/Omniwheel_config.yaml")
loader.override_with_args()
args = loader.get_config()

# General MuJoCo XML structure
mjcf = ET.Element("mujoco")
compiler = ET.SubElement(mjcf, "compiler", angle="radian")
option = ET.SubElement(mjcf, "option", integrator="implicitfast", timestep="0.002")
default = ET.SubElement(mjcf, "default")
default_omniwheel = ET.SubElement(default, "default", attrib={"class":"omniwheel"})
default_omniplate = ET.SubElement(default_omniwheel, "default", attrib={"class":"omniplate"})
default_omniroller = ET.SubElement(default_omniwheel, "default", attrib={"class":"omniroller"})
default_roller_geom = ET.SubElement(default_omniroller, 'geom', condim=f"{args['roller']['geom_condim']}",contype="1", conaffinity="0", friction=f"{args['roller']['geom_friction'][0]} {args['roller']['geom_friction'][1]} {args['roller']['geom_friction'][2]}")
worldbody = ET.SubElement(mjcf, "worldbody")

# wheel params
wr = args["wheel"]["radius"]
side = args["side"]

# plate params
px, py, pz = args["plate"]["pos"]
pr, ph = args["plate"]["size"]

# roller params
roller_r = wr * args["roller"]["ratio"]

# Generate the plate
plate_body = ET.SubElement(worldbody, "body", name=f"{side}_{args['plate']['name']}", pos=f"{px} {py} {pz}")
plate_inertia = ET.SubElement(plate_body, "inertial", pos="0 0 0", quat="0.707107 0 0 0.707107", mass=f"{args['plate']['mass']}", diaginertia=f"{args['plate']['diag_inertia'][0]} {args['plate']['diag_inertia'][1]} {args['plate']['diag_inertia'][2]}")
plate_joint = ET.SubElement(plate_body, "joint", name=f"{side}_{args['plate']['name']}", pos="0 0 0", axis="0 1 0")
plate_geom = ET.SubElement(plate_body, "geom", size=f"{pr} {ph}", quat="0.707107 0.707107 0 0", type="cylinder", rgba="0.2 0.2 0.2 0.5", contype="0", conaffinity="0")

# Generate the rollers
step = (2 * np.pi)/args["roller"]["n_roller"]

for i in range(args["roller"]["n_roller"]):
    body_name = f"{side}_{args['roller']['name']}{i}"
    joint_name = f"{side}_{args['roller']['name']}{i}"
    pin1 = np.array([
        (wr - roller_r) * np.cos(step * i),
        -ph / 2,
        (wr - roller_r) * np.sin(step * i)
    ])
    pin2 = np.array([
        (wr - roller_r) * np.cos(step * (i + 1)),
        ph / 2,
        (wr - roller_r) * np.sin(step * (i + 1))
    ])
    # Axis tangent to the plate at pin1
    axis = np.array([np.sin(step * i), 0, -np.cos(step * i)])
    pos = (pin1 + pin2)/2
    
    roller_body = ET.SubElement(plate_body, "body", name=body_name, pos=f"{pos[0]} {pos[1]} {pos[2]}")
    roller_inertia = ET.SubElement(roller_body, "inertial", pos="0 0 0", quat="0.707107 0 0 0.707107", mass=f"{args['roller']['mass']}", diaginertia=f"{args['roller']['diag_inertia'][0]} {args['roller']['diag_inertia'][1]} {args['roller']['diag_inertia'][2]}")
    roller_joint = ET.SubElement(roller_body, "joint", name=joint_name, pos="0 0 0", axis=f"{axis[0]} {axis[1]} {axis[2]}")
    roller_geom = ET.SubElement(roller_body, "geom", attrib={"type":"sphere","size":f"{roller_r}","class":"omniroller"},rgba="0.2 0.2 0.2 1")

# Pretty print the xml
ET.indent(mjcf, space="    ")

# Save to xml file
tree = ET.ElementTree(mjcf)
tree.write(f"DRBA_pipeline/Omniwheel_Generation/{side}_omniwheel.xml",encoding="utf-8", xml_declaration=True)

