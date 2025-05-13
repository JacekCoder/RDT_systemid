1-
# 定义参数
mass = 0.1  # kg
a = 0.065   # m (短轴半径)
b = 0.078   # m (长轴半径)

# 计算旋转轴向惯性
Iz = 0.5 * mass * np.power(a, 2)

# 计算垂直轴向惯性
Ixy = (1 / 5) * mass * (np.power(a, 2) + np.power(b, 2))

(f"旋转轴向惯性 Iz = {Iz:.8f} kg·m²")
(f"垂直轴向惯性 Ix=Iy = {Ixy:.8f} kg·m²") 
adjust diaginertia="0.00020618 0.00020618 0.00021125

2- geom friction调整为[0.4, 0.02,0.001]

3- ellipsoid code_gen

![image](https://github.com/user-attachments/assets/a00dda0d-f484-4261-9a52-54ea13310f16)
