Firstly, I wrote a script to generate a brand new XML file to replace the original one, allowing me to modify the interface's position in the model. This approach successfully changed the initial position. However, I discovered that directly modifying the "pos" attribute of the links breaks the assembled relationship between them. As a result, although the initial positions are updated, the links are no longer connected correctly when the simulation starts.

To solve this issue, I developed another script that uses inverse kinematics. When you input the desired interface position via the terminal, the script computes the corresponding joint configurations (qpos) for the following joints:
    "interface_ty",
    "L_distal", "L_fore", "L_toInterface",
    "R_distal", "R_fore", "R_toInterface"
 
 for example  can run the script like this:
 
 python3 arm.py -0.01754 -0.1 0.0
 
The script then checks whether the calculated joint positions produce an end-effector position within an acceptable error margin of the target. If the result is valid, it saves the corresponding qpos values for future reproducibility.

