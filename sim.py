import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0, 30.0, 30.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002
task_time: int = 2 # Move to the next task every 5 seconds

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

def grip_control (data, task) -> int:
    # Zero by default is closed, one is open
    if task == 'grasp':
        val = 150
    else:
        val = 0
    return val
    
def get_task_pose(model, data, task: str) -> np.ndarray:
    task_pose = np.zeros(7) # first 3 are positions, last 4 are quaternions
    if task == 'pre-grasp':
        task_pose[:3] = data.body("object").xpos
        task_pose[2] += 0.3 # the gripper should be 30cm above the target while pre-grasp in the z direction
        task_pose[3:] = data.body("object").xquat # set to the orientation of the target
    elif task == 'move-down':
        task_pose[:3] = data.body("object").xpos
        task_pose[2] += 0.15 # the gripper should be just above the box to grasp it. 
        task_pose[3:] = data.body("object").xquat # set to the orientation of the target
    elif task == 'grasp':
        task_pose[:3] = data.body("object").xpos
        task_pose[2] += 0.15 # the gripper should be just above the box to grasp it. 
        task_pose[3:] = data.body("object").xquat # set to the orientation of the target
    elif task == 'move-up':
        pass
    elif task == 'mocap':
        task_pose[:3] = data.body("target").xpos
        task_pose[3:] = data.body("target").xquat # set to the orientation of the target
    return task_pose

def execute_tasks(model, data, flag=1) -> None:
    """" (flag = 1) Core loop to cycle through task space and perform IK. (flag = 0) is for mocap mouse seek """

    if flag == 0:
        task_space = ['mocap']
    else:
        task_space = ['pre-grasp', 'move-down', 'grasp']
    
    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the DOF and actuator ids
    dof_ids = np.array([i for i in range(model.nv)])
    actuator_ids = dof_ids

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    # Start with a certain task in the task space. 
    task = task_space.pop(0) # the first run, this is 'pre-grasp'
    task_space.append(task)
    task_pose = get_task_pose(model, data, task)

    # Simulation counter. Move to the next task every (task_time / dt) ticks
    i = 0

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()

            # Spatial velocity (aka twist).
            task_pose = get_task_pose(model, data, task)
            #dx = named_body.xpos - data.site(site_id).xpos
            dx = task_pose[:3] - data.site(site_id).xpos
            #dx[2] += z_delta
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            #mujoco.mju_mulQuat(error_quat, named_body.xquat, site_quat_conj)
            mujoco.mju_mulQuat(error_quat, task_pose[3:], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, integration_dt)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Compute the task-space inertia matrix.
            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            # Compute generalized forces.
            tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

            # Add joint task in nullspace.
            Jbar = M_inv @ jac.T @ Mx
            ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]

            # Set the control signals
            arm_tau = tau[:7] # Only arm torques
            tau_combined = np.zeros(model.nu)
            tau_combined[:7] = arm_tau  # arm motors
            tau_combined[7]  = grip_control(data, task)
            np.clip(tau_combined, *model.actuator_ctrlrange.T, out=tau_combined)
            data.ctrl[:] = tau_combined

            # Step the simulation
            mujoco.mj_step(model, data)
            i += 1

            # Update the task every 2500 ticks
            if i % int(task_time / dt) == 0:
                task = task_space.pop(0) # the first run, this is 'pre-grasp'
                task_space.append(task)
                print(f"Switching to task at target pose: {task, task_pose}")
                print(f"Current end-effector pose: {data.site(site_id).xpos, site_quat}")

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    cwd = Path(__file__).resolve().parent  # Parent directory path

    # Load the model and data.
    model_path = str(cwd) + "/a3c_mujoco/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    execute_tasks(model, data, 1)
