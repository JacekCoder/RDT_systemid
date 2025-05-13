"""
This module is further complimented based on the public repo: mujoco_sysid（Author@Ivjonok）
Inertial parameters regressors are directly copied from the mujoco_sysid repo.
Joint damping and friction loss regressors are developed based on the MuJoCo inverse dynamics model (Author@rris-Wyf).
"""

import mujoco
import numpy as np
from numpy import typing as npt


def body_regressor(
    v_lin: npt.ArrayLike, v_ang: npt.ArrayLike, a_lin: npt.ArrayLike, a_ang: npt.ArrayLike
) -> npt.ArrayLike:
    """Y_body returns a regressor for a single rigid body

    Newton-Euler equations for a rigid body are given by:
    M * a_g + v x M * v = f

    where:
        M is the spatial inertia matrix of the body
        a_g is the acceleration of the body
        v is the spatial velocity of the body
        f is the spatial force acting on the body

    The regressor is a matrix Y such that:
        Y \theta = f

    where:
        \theta is the vector of inertial parameters of the body (10 parameters)

    More expressive derivation is given here:
        https://colab.research.google.com/drive/1xFte2FT0nQ0ePs02BoOx4CmLLw5U-OUZ?usp=sharing

    Args:
        v_lin (npt.ArrayLike): linear velocity of the body
        v_ang (npt.ArrayLike): angular velocity of the body
        a_lin (npt.ArrayLike): linear acceleration of the body
        a_ang (npt.ArrayLike): angular acceleration of the body

    Returns:
        npt.ArrayLike: regressor for the body
    """
    v1, v2, v3 = v_lin
    v4, v5, v6 = v_ang

    a1, a2, a3 = a_lin
    a4, a5, a6 = a_ang

    # fmt: off
    return np.array([
        [a1 - v2*v6 + v3*v5, -v5**2 - v6**2, -a6 + v4*v5, a5 + v4*v6, 0, 0, 0, 0, 0, 0],
        [a2 + v1*v6 - v3*v4, a6 + v4*v5, -v4**2 - v6**2, -a4 + v5*v6, 0, 0, 0, 0, 0, 0],
        [a3 - v1*v5 + v2*v4, -a5 + v4*v6, a4 + v5*v6, -v4**2 - v5**2, 0, 0, 0, 0, 0, 0],
        [0, 0, a3 - v1*v5 + v2*v4, -a2 - v1*v6 + v3*v4, a4, a5 - v4*v6, -v5*v6, a6 + v4*v5, v5**2 - v6**2, v5*v6],
        [0, -a3 + v1*v5 - v2*v4, 0, a1 - v2*v6 + v3*v5, v4*v6, a4 + v5*v6, a5, -v4**2 + v6**2, a6 - v4*v5, -v4*v6],
        [0, a2 + v1*v6 - v3*v4, -a1 + v2*v6 - v3*v5, 0, -v4*v5, v4**2 - v5**2, v4*v5, a4 - v5*v6, a5 + v4*v6, a6]
    ])
    # fmt: on


def joint_body_regressor(mj_model, mj_data, body_id) -> npt.ArrayLike:
    """mj_bodyRegressor returns a regressor for a single rigid body

    This function calculates the regressor for a single rigid body in the MuJoCo model.
    Given the index of body we compute the velocity and acceleration of the body and
    then calculate the regressor using the Y_body function.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        body_id: ID of the body

    Returns:
        npt.ArrayLike: regressor for the body
    """

    velocity = np.zeros(6)
    accel = np.zeros(6)
    _cross = np.zeros(3)

    mujoco.mj_objectVelocity(mj_model, mj_data, 2, body_id, velocity, 1)
    mujoco.mj_objectAcceleration(mj_model, mj_data, 2, body_id, accel, 1)

    v, w = velocity[3:], velocity[:3]
    # dv - classical acceleration, already contains g
    dv, dw = accel[3:], accel[:3]
    mujoco.mju_cross(_cross, w, v)

    # for floating base, this is already included in dv
    if mj_model.nq == mj_model.nv:
        dv -= _cross

    return body_regressor(v, w, dv, dw)


def get_jacobian(mjmodel, mjdata, bodyid):
    R = mjdata.xmat[bodyid].reshape(3, 3)

    jac_lin, jac_rot = np.zeros((3, mjmodel.nv)), np.zeros((3, mjmodel.nv))
    mujoco.mj_jacBody(mjmodel, mjdata, jac_lin, jac_rot, bodyid)

    return np.vstack([R.T @ jac_lin, R.T @ jac_rot])


def joint_inertial_torque_regressor(mj_model, mj_data) -> npt.ArrayLike:
    """mj_jointRegressor returns a regressor for the whole model

    This function calculates the regressor for the whole model in the MuJoCo model.

    This regressor is computed to use in joint-space calculations. It is a matrix that
    maps the inertial parameters of the bodies to the generalized forces.

    Newton-Euler equations for a rigid body are given by:
        M * a_g + v x M * v = f

    Expressing the spatial quantities in terms of the generalized quantities
    we can rewrite the equation for the system of bodies as:
        M * q_dot_dot + h = tau

    Where
        M is the mass matrix
        h is the bias term
        tau is the generalized forces

    Then, the regressor is a matrix Y such that:
        Y * theta = tau

    where:
        theta is the vector of inertial parameters of the bodies (10 parameters per body):
            theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]


    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        npt.ArrayLike: regressor for the whole model
    """

    njoints = mj_model.njnt
    body_regressors = np.zeros((6 * njoints, njoints * 10))
    col_jac = np.zeros((6 * njoints, mj_model.nv))

    for i, body_id in enumerate(mj_model.jnt_bodyid):
        # calculate cody regressors
        body_regressors[6 * i : 6 * (i + 1), 10 * i : 10 * (i + 1)] = joint_body_regressor(mj_model, mj_data, body_id)

        col_jac[6 * i : 6 * i + 6, :] = get_jacobian(mj_model, mj_data, body_id)

    return col_jac.T @ body_regressors

def joint_damping_torque_regressor(mj_model, mj_data) -> npt.ArrayLike:
    """
    This function calculates the regressor for joint damping in the MuJoCo model.
    Damping regressor is diagonal matrix with digonal elements as joint velocities (qvel).
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        npt.ArrayLike: regressor for joint damping
    """
    njoints = mj_model.njnt

    return -np.diag(-mj_data.qvel)

def joint_frictionloss_torque_regressor(mj_model, mj_data) -> npt.ArrayLike:
    """
    This function calculates the regressor for joint friction loss in the MuJoCo model.
    Friction loss regressor is diagonal matrix.
    As friction loss (static friction) is modeled with Huber "norm" -
    The digonal elements are the solution of a quadratic optimization with box constraints (floss <=|mj_model.dof_frictionloss|)
    (simplied when only friction loss constraints are considered)
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data

    Returns:
        npt.ArrayLike: regressor for joint friction loss
    """
    
    """
    To Do:
    1. Current representation is for nonlinear friction regressor use.
    2. If not required, the simpliest rerpesentation is efc_force/dof_frictionloss
    """
    
    njoints = mj_model.njnt
    joint_frictionloss_regressors = np.zeros((njoints, njoints))

    # jar = Jac@qacc - efc_aref(-constraint_damping*qvel)
    efc_damping = np.array([mj_data.efc_KBIP[i,1].copy() for i in range(njoints)])
    jar = np.zeros(njoints)
    jar = mj_data.qacc.copy()-(-efc_damping*mj_data.qvel.copy())
    
    # compare jar/efc_R with -1 and 1
    return -np.diag(np.clip(-jar/(mj_data.efc_R*mj_model.dof_frictionloss), -1, 1))
