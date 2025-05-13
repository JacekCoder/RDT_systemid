import numpy as np
import xml.etree.ElementTree as ET

# MJCF structure setup
mjcf = ET.Element("mujoco")
worldbody = ET.SubElement(mjcf, "worldbody")

# Parameters
plate_radius = 0.13
plate_height = 0.0435
plate_mass = 2.2423
plate_inertia = [0.0524193, 0.0303095, 0.0303095]
roller_mass = 0.1
roller_inertia = [0.00020618, 0.00020618, 0.00021125]
default_roller_size = [0.09, 0.04, 0.04]
n_rollers = 12

# Plate definition
plate_body = ET.SubElement(worldbody, "body", name="plate_body", pos="0 0 0")
ET.SubElement(plate_body, "inertial", pos="0 0 0", quat="0.707107 0 0 0.707107", mass=f"{plate_mass}", diaginertia=f"{plate_inertia[0]} {plate_inertia[1]} {plate_inertia[2]}")
ET.SubElement(plate_body, "joint", name="plate_joint", pos="0 0 0", axis="0 1 0")
ET.SubElement(plate_body, "geom", size=f"{plate_radius} {plate_height}", quat="0.707107 0.707107 0 0", type="cylinder", rgba="0.2 0.2 0.2 0.5", contype="0", conaffinity="0")

# Rollers definition
step_angle = (2 * np.pi) / n_rollers
roller_offset_radius = plate_radius - default_roller_size[1]

for i in range(n_rollers):
    angle = step_angle * i
    pos_x = roller_offset_radius * np.cos(angle)
    pos_z = roller_offset_radius * np.sin(angle)

    axis_x = -np.sin(angle)
    axis_z = np.cos(angle)

    roller_body = ET.SubElement(plate_body, "body", name=f"roller_{i}", pos=f"{pos_x:.6f} 0 {pos_z:.6f}")
    ET.SubElement(roller_body, "inertial", pos="0 0 0", quat="0.707107 0 0 0.707107", mass=f"{roller_mass}", diaginertia=f"{roller_inertia[0]} {roller_inertia[1]} {roller_inertia[2]}")
    ET.SubElement(roller_body, "joint", name=f"roller_joint_{i}", pos="0 0 0", axis=f"{axis_x:.6f} 0 {axis_z:.6f}")

    # Set special size and color for roller_0
    if i == 0:
        roller_size = [0.07, 0.045, 0.045]
        roller_rgba = "1 1 1 1"
    else:
        roller_size = default_roller_size
        roller_rgba = "0.2 0.2 0.2 1"

    euler_angle = 90 - i * 30  # Decrease angle by 30 degrees each roller

    ET.SubElement(
        roller_body, "geom",
        attrib={
            "type": "ellipsoid",
            "size": f"{roller_size[0]} {roller_size[1]} {roller_size[2]}",
            "rgba": roller_rgba,
            "euler": f"0 {euler_angle} 0"
        }
    )

# Pretty print the XML
ET.indent(mjcf, space="    ")

# Save to XML file
ET.ElementTree(mjcf).write("ellipsoid_omniwheel.xml", encoding="utf-8", xml_declaration=True)

print("Omniwheel with ellipsoid rollers MJCF file generated successfully.")
