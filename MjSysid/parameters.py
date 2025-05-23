import mujoco
import numpy as np
import numpy.typing as npt


def skew(vector):
    return np.cross(np.eye(vector.size), vector.reshape(-1))


def get_inertial_parameters(mjmodel, body_id) -> npt.ArrayLike:
    """Get the inertial parameters \theta of a body
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body

    Returns:
        npt.ArrayLike: theta of the body
    """
    mass = mjmodel.body(body_id).mass[0]
    rc = mjmodel.body(body_id).ipos
    diag_inertia = mjmodel.body(body_id).inertia

    # get the orientation of the body
    r_flat = np.zeros(9)
    mujoco.mju_quat2Mat(r_flat, mjmodel.body(body_id).iquat)

    R = r_flat.reshape(3, 3)

    shift = mass * skew(rc) @ skew(rc)
    mjinertia = R @ np.diag(diag_inertia) @ R.T - shift

    upper_triangular = np.array(
        [
            mjinertia[0, 0],
            mjinertia[0, 1],
            mjinertia[1, 1],
            mjinertia[0, 2],
            mjinertia[1, 2],
            mjinertia[2, 2],
        ]
    )

    return np.concatenate([[mass], mass * rc, upper_triangular])

def get_damping_parameters(mjmodel,jnt_id) -> npt.ArrayLike:
    """Get the damping parameters of a joint

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        jnt_id (int): The id of the joint

    Returns:
        npt.ArrayLike: damping coefficients
    """
    return np.array(mjmodel.dof_damping[jnt_id])

def get_frictionloss_parameters(mjmodel,jnt_id) -> npt.ArrayLike:
    """Get the frictionloss parameters of a joint

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        jnt_id (int): The id of the joint

    Returns:
        npt.ArrayLike: fricionloss
    """
    return np.array(mjmodel.dof_frictionloss[jnt_id])

def set_inertial_parameters(mjmodel, body_id, theta: npt.ArrayLike) -> None:
    """Set the dynamic parameters to a body

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body
        theta (npt.ArrayLike): The dynamic parameters of the body
    """

    mass = theta[0]
    rc = theta[1:4] / mass
    inertia = theta[4:]
    inertia_full = np.array(
        [
            [inertia[0], inertia[1], inertia[3]],
            [inertia[1], inertia[2], inertia[4]],
            [inertia[3], inertia[4], inertia[5]],
        ]
    )

    # shift the inertia
    inertia_full += mass * skew(rc) @ skew(rc)

    # eigen decomposition
    eigval, eigvec = np.linalg.eigh(inertia_full)
    R = eigvec
    diag_inertia = eigval

    # check if singular, then abort
    if np.any(np.isclose(diag_inertia, 0)):
        raise ValueError("Cannot deduce inertia matrix because RIR^T is singular.")

    # set the mass
    mjmodel.body(body_id).mass = np.array([mass])
    mjmodel.body(body_id).ipos = rc

    # set the orientation
    mujoco.mju_mat2Quat(mjmodel.body(body_id).iquat, R.flatten())

    # set the inertia
    mjmodel.body(body_id).inertia = diag_inertia

def set_damping_parameters(mjmodel,jnt_id,thetaD: npt.ArrayLike) -> None:
    """Set the damping parameters of a joint

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        jnt_id (int): The id of the joint
        theta (npt.ArrayLike): The damping parameters
    """
    mjmodel.dof_damping[jnt_id] = thetaD

def set_frictionloss_parameters(mjmodel,jnt_id,thetaF: npt.ArrayLike) -> None:
    """Set the frictionloss parameters of a joint

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        jnt_id (int): The id of the joint
        theta (npt.ArrayLike): The frictionloss parameters
    """
    mjmodel.dof_frictionloss[jnt_id] = thetaF