import mujoco
from dm_control import mjcf

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from .math import quat2rotm, quat_inverse, quat_product
import jax

@jax.jit
def apply_ndi_control(
    state_qpos, 
    state_qvel, 
    action, 
    drone_masses, 
    drone_inertias,
    qpos_adrs,
    dof_adrs
):
    g_vec = jnp.array([0.0, 0.0, -9.81])

    # controller parameters from GeometricController
    kp_att_xy = 150.0
    kp_att_z = 5.0
    kp_rate = jnp.array([25.0, 25.0, 8.0])

    kappa = 0.022
    beta = jnp.pi / 4
    l = 0.10606601717798213

    G_1 = jnp.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [
                l * jnp.sin(beta),
                -l * jnp.sin(beta),
                -l * jnp.sin(beta),
                l * jnp.sin(beta),
            ],
            [
                -l * jnp.cos(beta),
                -l * jnp.cos(beta),
                l * jnp.cos(beta),
                l * jnp.cos(beta),
            ],
            [kappa, -kappa, kappa, -kappa],
        ]
    )
    G_1_inv = jnp.linalg.inv(G_1)

    thrust_min_collective = 0.0
    thrust_max_collective = 6.25 * 4
    thrust_min = 0.0
    thrust_max = 6.25

    epsilon = 1e-6
    action_reshaped = action.reshape(-1, 6)

    def single_drone_control(m, inertia_diag, qpos_adr, dof_adr, act):
        inertia_mat = jnp.diag(inertia_diag)
        quat = jax.lax.dynamic_slice_in_dim(state_qpos, qpos_adr + 3, 4)
        ang_vel_body = jax.lax.dynamic_slice_in_dim(state_qvel, dof_adr + 3, 3)

        des_acc = act[0:3]
        omega_b_ref = act[3:6]

        acc_cmd = des_acc - g_vec
        
        collective_thrust_des_magnitude = jnp.linalg.norm(acc_cmd) * m
        T_cmd = jnp.clip(collective_thrust_des_magnitude, thrust_min_collective, thrust_max_collective)
        
        z_b_des = acc_cmd / (jnp.linalg.norm(acc_cmd) + epsilon)
        
        yaw_current = jnp.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        x_intermediate_des = jnp.array([jnp.cos(yaw_current), jnp.sin(yaw_current), 0.0])
        
        y_b_des_unnormalized = jnp.cross(z_b_des, x_intermediate_des)
        y_b_des = y_b_des_unnormalized / (jnp.linalg.norm(y_b_des_unnormalized) + epsilon)
        x_b_des = jnp.cross(y_b_des, z_b_des)
        
        R_des = jnp.column_stack([x_b_des, y_b_des, z_b_des])
        
        tr = R_des[0,0] + R_des[1,1] + R_des[2,2]
        q_cmd_w = jnp.sqrt(jnp.maximum(1.0 + tr, 0.0)) / 2.0
        q_cmd_x = (R_des[2,1] - R_des[1,2]) / (4.0 * q_cmd_w + epsilon)
        q_cmd_y = (R_des[0,2] - R_des[2,0]) / (4.0 * q_cmd_w + epsilon)
        q_cmd_z = (R_des[1,0] - R_des[0,1]) / (4.0 * q_cmd_w + epsilon)
        q_cmd = jnp.array([q_cmd_w, q_cmd_x, q_cmd_y, q_cmd_z])
        q_cmd = q_cmd / (jnp.linalg.norm(q_cmd) + epsilon)

        q_inv = quat_inverse(quat)
        q_diff = quat_product(q_inv, q_cmd)

        q_e_w, q_e_x, q_e_y, q_e_z = q_diff[0], q_diff[1], q_diff[2], q_diff[3]
        
        norm_factor = 2.0 / (jnp.sqrt(q_e_w**2 + q_e_z**2) + epsilon)
        
        q_e_red = norm_factor * jnp.array([
            q_e_w * q_e_x - q_e_y * q_e_z,
            q_e_w * q_e_y + q_e_x * q_e_z,
            0.0
        ])
        q_e_yaw = norm_factor * jnp.array([0.0, 0.0, q_e_z])
        
        alpha_b_des = (
            kp_att_xy * q_e_red
            + kp_att_z * jnp.sign(q_e_w) * q_e_yaw
            + kp_rate * (omega_b_ref - ang_vel_body)
        )

        moments = jnp.dot(inertia_mat, alpha_b_des) + jnp.cross(ang_vel_body, jnp.dot(inertia_mat, ang_vel_body))
        mu_ndi = jnp.array([T_cmd, moments[0], moments[1], moments[2]])
        
        thrusts = jnp.dot(G_1_inv, mu_ndi)
        thrusts = jnp.clip(thrusts, thrust_min, thrust_max)
        
        return thrusts

    thrusts_all = jax.vmap(single_drone_control)(
        drone_masses, drone_inertias, qpos_adrs, dof_adrs, action_reshaped
    )
    return thrusts_all.flatten()


def build_multi_agent_scene(
    target_xml_path,
    num_cars,
    enable_payload_actuator=False,
    payload_force_limit=80.0,
):
    # Load the base static scene
    model_path = Path(__file__).parent / "assets" / "scene.xml"
    scene = mjcf.from_path(model_path)

    # Load and attach the target object
    target_payload = mjcf.from_path(target_xml_path)
    target_frame = scene.attach(target_payload)
    payload_freejoint = target_frame.add("freejoint", name="payload_freejoint")
    target_frame.pos = [0, 0, 0.5]

    if enable_payload_actuator:
        scene.actuator.add(
            "motor",
            name="payload_fx",
            joint=payload_freejoint,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=True,
            ctrlrange=[-payload_force_limit, payload_force_limit],
        )
        scene.actuator.add(
            "motor",
            name="payload_fy",
            joint=payload_freejoint,
            gear=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=True,
            ctrlrange=[-payload_force_limit, payload_force_limit],
        )
        scene.actuator.add(
            "motor",
            name="payload_fz",
            joint=payload_freejoint,
            gear=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ctrllimited=True,
            ctrlrange=[-payload_force_limit, payload_force_limit],
        )

    # Load and attach a ghost target for visualizing the target pose
    ghost_payload = mjcf.from_path(target_xml_path)
    for geom in ghost_payload.find_all("geom"):
        geom.conaffinity = 0
        geom.contype = 0
        if geom.rgba is not None:
            geom.rgba = [geom.rgba[0], geom.rgba[1], geom.rgba[2], 0.3]
        else:
            geom.rgba = [0.5, 0.5, 0.5, 0.3]

    # We remove any hooks/sites from the ghost to avoid clutter
    for site in ghost_payload.find_all("site"):
        site.remove()

    ghost_frame = scene.attach(ghost_payload)
    ghost_frame.mocap = "true"

    # Default rope_end offset in drone local frame (from drone.xml kinematic chain).
    # Spawning each drone at hook_pos + this offset starts connect constraints near zero error.
    rope_end_offset_z = 1.05

    # Dynamically find how many hooks the loaded target has
    hook_sites = [
        site for site in target_payload.find_all("site") if "hook_" in site.name
    ]
    num_drones = len(hook_sites)

    # Spawn drones and connect them to the hooks
    for i in range(num_drones):
        drone_path = Path(__file__).parent / "assets" / "drone.xml"
        drone = mjcf.from_path(drone_path)

        # Grab the exact hook site from the payload
        target_hook = target_payload.find("site", f"hook_{i}")
        hook_local = np.array(target_hook.pos, dtype=float)
        payload_world = np.array(target_frame.pos, dtype=float)
        hook_world = payload_world + hook_local

        # Spawn directly above the matching hook to satisfy rope connect constraints at start.
        spawn_pos = [
            float(hook_world[0]),
            float(hook_world[1]),
            float(hook_world[2] + rope_end_offset_z),
        ]
        drone_frame = scene.attach(drone)
        drone_frame.add("freejoint")
        drone_frame.pos = spawn_pos

        # Grab the rope end from this specific drone
        drone_rope = drone.find("site", "rope_end")

        # Link them dynamically
        scene.equality.add(
            "connect",
            site1=drone_rope,
            site2=target_hook,
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.001],
        )

    # Spawn the grounded vehicles
    for j in range(num_cars):
        car_path = Path(__file__).parent / "assets" / "omnicar.xml"
        car = mjcf.from_path(car_path)
        spawn_pos = [2, (j * 1.5) - ((num_cars - 1) * 0.75), 0.15]
        car_frame = scene.attach(car)
        car_frame.pos = spawn_pos

    return scene


def _instance_sort_key(prefix):
    """Sort instance names like drone, drone_1, drone_2 in natural order."""
    if "_" not in prefix:
        return (prefix, 0)
    base, maybe_idx = prefix.rsplit("_", 1)
    if maybe_idx.isdigit():
        return (base, int(maybe_idx))
    return (prefix, 0)


def get_robot_indices(model):
    """Build actuator/state index maps for all drone and omnicar instances."""
    drone_ctrl_by_prefix = {}
    car_ctrl_by_prefix = {}

    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        if not act_name or "/" not in act_name:
            continue
        prefix, actuator = act_name.split("/", 1)

        if actuator.startswith("thrust"):
            drone_ctrl_by_prefix.setdefault(prefix, {})[actuator] = act_id
        elif actuator in ("move_x", "move_y", "turn_yaw"):
            car_ctrl_by_prefix.setdefault(prefix, {})[actuator] = act_id

    drone_state_by_prefix = {}
    car_state_by_prefix = {}

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name:
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])

        if "/" in joint_name:
            prefix, joint = joint_name.split("/", 1)
        else:
            prefix, joint = joint_name, ""

        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE) and prefix.startswith("drone"):
            drone_state_by_prefix[prefix] = {
                "qpos_adr": qpos_adr,
                "dof_adr": dof_adr,
            }

        if joint == "omni_x" and prefix.startswith("omnicar"):
            car_state_by_prefix[prefix] = {
                "qpos_adr": qpos_adr,
                "dof_adr": dof_adr,
            }

    drones = []
    for prefix in sorted(drone_ctrl_by_prefix.keys(), key=_instance_sort_key):
        ctrl = drone_ctrl_by_prefix[prefix]
        state = drone_state_by_prefix.get(prefix)
        if not state:
            continue
        if not all(k in ctrl for k in ("thrust1", "thrust2", "thrust3", "thrust4")):
            continue

        drones.append(
            {
                "name": prefix,
                "ctrl": [
                    ctrl["thrust1"],
                    ctrl["thrust2"],
                    ctrl["thrust3"],
                    ctrl["thrust4"],
                ],
                "qpos_adr": state["qpos_adr"],
                "dof_adr": state["dof_adr"],
            }
        )

    cars = []
    for prefix in sorted(car_ctrl_by_prefix.keys(), key=_instance_sort_key):
        ctrl = car_ctrl_by_prefix[prefix]
        state = car_state_by_prefix.get(prefix)
        if not state:
            continue
        if not all(k in ctrl for k in ("move_x", "move_y", "turn_yaw")):
            continue

        cars.append(
            {
                "name": prefix,
                "ctrl": [ctrl["move_x"], ctrl["move_y"], ctrl["turn_yaw"]],
                "qpos_adr": state["qpos_adr"],
                "dof_adr": state["dof_adr"],
            }
        )

    return {"drones": drones, "cars": cars}

