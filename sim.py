import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002
task_time: int = 5 # Move to the next task every 5 seconds

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-5

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0

def get_task_pose(model, data, task: str) -> np.ndarray:
    task_pose = np.zeros(7) # first 3 are positions, last 4 are quaternions
    if task == 'pre-grasp':
        task_pose[:3] = data.body("object").xpos
        task_pose[2] += 0.3 # the gripper should be 30cm above the target while pre-grasp in the z direction
        task_pose[3:] = data.body("object").xquat # set to the orientation of the target
    elif task == 'move-down':
        task_pose[:3] = data.body("object").xpos
        task_pose[2] += 0.15 # the gripper should be just above the object to grasp it. 
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

def plot_errors(error_matrix, time_steps) -> None:
    # Convert the error list to a NumPy array of shape (num_steps, 6)
    error_matrix = np.array(error_matrix)

    # Plot each of the 6 error components over time.
    plt.figure(figsize=(10, 6))
    for j in range(6):
        plt.plot(time_steps, error_matrix[:, j], label=f"Error component {j}")

    plt.xlabel("Time (s)")
    plt.ylabel("Error Value")
    plt.title("6D Error Components vs. Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("error_plot.png")
    print("Error plot saved as 'error_plot.png'.")

def execute_tasks(model, data, flag=1) -> None:
    """" (flag = 1) Core loop to cycle through task space and perform IK. (flag = 0) is for mocap mouse seek """

    if flag == 0:
        task_space = ['mocap']
    else:
        task_space = ['pre-grasp']
    
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
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Simulation counter. Move to the next task every (task_time / dt) ticks
    i = 0

    # Initialize lists to record time and the 6D error vector at each timestep.
    time_steps = []
    error_matrix = []  # Each element will be a 6-element vector (list or array)

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

            # Start with a certain task in the task space. 
            task = task_space[0]
            task_pose = get_task_pose(model, data, task)

            #dx = named_body.xpos - data.site(site_id).xpos
            error_pos[:] = task_pose[:3] - data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, task_pose[3:], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, integration_dt)

            #print(f"Task: {task}, Error: {error}")
            # Record the current time (simulation step count * dt) and the full error vector.
            time_steps.append(i * dt)
            error_matrix.append(error.copy())  # Save a copy of the current 6-element error vector.

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation
            mujoco.mj_step(model, data)
            i += 1

            # Update the task every X ticks
            if i % int(task_time / dt) == 0:
                task = task_space.pop(0) # the first run, this is 'pre-grasp'
                task_space.append(task)
                error_integral = np.zeros(6)
                print(f"Switching to task at target pose: {task, task_pose}")

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    
    viewer.close()
    # Save error plot
    plot_errors(error_matrix, time_steps)

if __name__ == "__main__":
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    cwd = Path(__file__).resolve().parent  # Parent directory path

    # Load the model and data.
    model_path = str(cwd) + "/a3c_mujoco/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    execute_tasks(model, data, 1)
