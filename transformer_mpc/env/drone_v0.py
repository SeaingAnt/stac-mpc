"""Drone V0 Environment.
JAX implementation of the Mikrokopter drone environment considering uncertainties on the mass, inertia, 
position of the center of mass.
"""

from functools import partial
import os
import sys
from typing import Any, List
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import TEnvParams, TEnvState

from .math import quat2rotm, quat_product, skew, quat_inverse

@struct.dataclass
class ModelState:
    pos: jax.Array
    attitude: jax.Array  # (unit quaternion [w, x, y, z])
    vel: jax.Array
    omega: jax.Array


@struct.dataclass
class EnvState(environment.EnvState):
    state: ModelState
    target_pos: jax.Array
    target_yaw: jax.Array  # Desired yaw angle (rotation around z-axis)
    last_u: jax.Array  # Only needed for rendering
    time: int

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self.state, name)


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0, -1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0, 1.0])
    )

    min_input: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([4.0, -3.0, -3.0, -1.0]),
    )
    max_input: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([25.0, 3.0, 3.0, 1.0])
    )
    env_max_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([1.0, 1.0, 1.0])
    )
    env_min_limits: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([-1.0, -1.0, 0.0])
    )
    dt: float = struct.field(pytree_node=False, default=0.02)
    g: float = struct.field(pytree_node=False, default=9.81)
    m: float = 1.4
    I: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([0.022, 0.022, 0.020]),
    )
    com_pos: jax.Array = struct.field(
        pytree_node=False, default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    delay: float = 0.002

    max_steps_in_episode: int = struct.field(pytree_node=False, default=200)
    is_testing: bool = struct.field(pytree_node=False, default=False)
    mpc_horizon: int = struct.field(pytree_node=False, default=10)


class DroneV0(environment.Environment):
    """JAX Compatible version of mikrokopter drone converging to a target."""

    def __init__(self):
        self.type = ""
        self.state_dim = 13
        self.obs_dim = 14  # Added yaw_error to observations
        self.action_dim = self.action_space(self.default_params).shape[0]
        self.param_dim = 8  # len(jax.tree.leaves(self.default_params))
        self.target_pos = jnp.array([0.0, 0.0, 0.5])
        self.hovering_input = jnp.array(
            [self.default_params.m * self.default_params.g, 0.0, 0.0, 0.0]
        )

        self.params_min = [
            1.0,  # mass
            jnp.array([0.01, 0.01, 0.01]),  # inertia
            jnp.array([-0.02, -0.02, -0.02]),  # com_pos
            0.002,  # delay
        ]
        self.params_max = [
            1.8,  # mass
            jnp.array([0.05, 0.05, 0.05]),  # inertia
            jnp.array([0.02, 0.02, 0.02]),  # com_pos
            0.01,  # delay
        ]
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Drone-v0."""
        return EnvParams()
    
    class mpc_dynamics:
        def __init__(self, state_dim, num_actions, env_params):
            self.static_m = env_params.m
            self.static_g = env_params.g
            self.static_I = jnp.asarray(env_params.I, dtype=jnp.float32)
            self.static_com_pos = jnp.asarray(env_params.com_pos, dtype=jnp.float32)
            self.static_min_input = jnp.asarray(env_params.min_input, dtype=jnp.float32)
            self.static_max_input = jnp.asarray(env_params.max_input, dtype=jnp.float32)

        def state_dot(self, state, control, params):
            m = self.static_m
            g = self.static_g
            I = self.static_I
            com_pos = self.static_com_pos
            min_input = self.static_min_input
            max_input = self.static_max_input

            quat = state[3:7]
            omega = state[10:13]
            quat = quat / jnp.sqrt(jnp.sum(quat**2) + 1e-6)

            u = control # Removed clipping to ensure smooth gradients

            orientation_mat = quat2rotm(quat)
            ang_vel_skew = skew(omega)

            total_force = (
                jnp.array([0.0, 0.0, u[0]])
                - m * g * orientation_mat[2, :]
                - m * ang_vel_skew @ ang_vel_skew @ com_pos
            )
            total_torque = (
                u[1:4]
                - ang_vel_skew @ jnp.diag(I) @ omega
                - m * g * skew(com_pos) @ orientation_mat[2, :]
            )

            si_mat = jnp.diag(jnp.concatenate([m * jnp.ones(3, dtype=jnp.float32), I]))
            si_mat = si_mat.at[0:3, 3:6].set(-m * skew(com_pos))
            si_mat = si_mat.at[3:6, 0:3].set(m * skew(com_pos))

            acc = jnp.linalg.solve(si_mat, jnp.concatenate([total_force, total_torque]))
            acc_lin = orientation_mat @ acc[:3]
            acc_ang = acc[3:6]

            pos_dot = state[7:10]
            omega_quat = jnp.array([0.0, omega[0], omega[1], omega[2]])
            quat_dot = 0.5 * quat_product(quat, omega_quat)

            return jnp.concatenate([pos_dot, quat_dot, acc_lin, acc_ang])

    def rescale_action(self, action: jax.Array, params: EnvParams) -> jax.Array:
        action_span = params.max_action - params.min_action
        input_span = params.max_input - params.min_input
        normalized = (action - params.min_action) / action_span
        return params.min_input + normalized * input_span

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams | None = None,
    ) -> tuple[jax.Array, TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Overload step to remove the reset when done."""

        if params is None:
            params = self.default_params

        # Step
        key_step, _ = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        return obs_st, state_st, reward, done, info

    @partial(jax.jit, static_argnums=(0))
    def step_env(
        self,
        key: jax.Array,
        env_state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Integrate pendulum ODE and return transition."""

        action_clipped = jnp.clip(action, params.min_action, params.max_action)
        action_rescaled = self.rescale_action(action_clipped, params)

        state_next = self.step_dynamics(key, env_state, action_rescaled, params)

        pos_diff_norm = jnp.linalg.norm(state_next.pos - env_state.target_pos)

        # Main task score: keep this non-saturating so reward does not collapse to ~0 far from target.
        pos_score = jnp.exp(-2.0 * pos_diff_norm)
        close_bonus = jax.lax.select(
            pos_diff_norm < 0.02,
            4.0,
            0.0,
        )

        # Compute yaw error from quaternion
        # Yaw can be extracted from quaternion as: yaw = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2))
        q = state_next.attitude
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.abs(jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                         jnp.cos(current_yaw - env_state.target_yaw)))

        reward_yaw = jax.lax.select(
            yaw_error < 0.05,
            1.0,  # reward when close to target
            jnp.exp(-2.0 * yaw_error),
        )

        thrust_normalized_input = action_rescaled.at[0].set(action_rescaled[0] / (params.max_input[0]))
        thrust_normalized_input_last = env_state.last_u.at[0].set(env_state.last_u[0] / (params.max_input[0]))

        omega_score = jnp.exp(-1.0 * jnp.linalg.norm(state_next.omega))
        hover_score = jnp.exp(
            -1.5
            * jnp.linalg.norm(
                thrust_normalized_input - self.hovering_input / params.max_input[0]
            )
        )
        smooth_score = jnp.exp(
            -1.5 * jnp.linalg.norm(thrust_normalized_input - thrust_normalized_input_last)
        )

        # Extra damping around the goal to reduce limit-cycle oscillations.
        weighted_vel = jnp.array(
            [state_next.vel[0], state_next.vel[1], 1.8 * state_next.vel[2]]
        )
        vel_score = jnp.exp(-2.5 * jnp.linalg.norm(weighted_vel))
        near_target_gate = jnp.exp(-8.0 * pos_diff_norm)
        target_stability_score = near_target_gate * vel_score * omega_score
        target_settle_score = near_target_gate * smooth_score

        # Position-gated weighted sum: informative reward scale, but no high reward while drifting/falling.
        reward = (
            3.0 * pos_score
            + 1.0 * pos_score*reward_yaw
            + 0.6 * reward_yaw *pos_score* omega_score
            + 0.4 * pos_score * hover_score
            + 0.2 * pos_score * smooth_score
            + 1.2 * target_stability_score
            + 0.8 * target_settle_score
            + close_bonus
        )

        reward = reward.squeeze()

        env_state = EnvState(
            state=state_next,
            target_pos=env_state.target_pos,
            target_yaw=env_state.target_yaw,
            last_u=action_rescaled.reshape(-1),
            time=env_state.time + 1,
        )

        done = self.is_terminal(env_state, params)
        return (
            self.get_obs(env_state),
            env_state,
            reward,
            done,
            {"discount": self.discount(env_state, params)},
        )


    def step_dynamics(
        self,
        key: jax.Array,
        env_state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:
        """Step dynamics with input delay."""

        state = env_state.state
        last_action = env_state.last_u
        
        dt1 = jnp.clip(params.delay, 0.0, params.dt)
        dt2 = params.dt - dt1

        params = params.replace(dt=dt1)
        state_intermediate = self._step_dynamics(key, state, last_action, params)
        params = params.replace(dt=dt2)
        state_final = self._step_dynamics(key, state_intermediate, action, params)

        return state_final

    def _step_dynamics(
        self,
        key: jax.Array,
        state: ModelState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> ModelState:

        u = jnp.clip(action, params.min_input, params.max_input)

        quat = state.attitude
        ang_vel = state.omega
        ang_vel_skew = skew(ang_vel)

        orientation_mat = quat2rotm(quat)

        total_force = (
            jnp.array([0.0, 0.0, u[0]]) - params.m * params.g * orientation_mat[2, :] - params.m * ang_vel_skew @ ang_vel_skew @ params.com_pos
        )  # transpose + 3rd col = 3rd row

        total_torque = (
            u[1:4] - ang_vel_skew @ params.I @ ang_vel - params.m * params.g * skew(params.com_pos) @ orientation_mat[2, :]
        ) 

        # Spatial Inertia Matrix
        si_mat = jnp.diag(
            jnp.concatenate([params.m * jnp.ones(3, dtype=jnp.float32), params.I])
        )
        # Off diagonal terms due to CoM offset
        si_mat = si_mat.at[0:3, 3:6].set(-params.m * skew(params.com_pos))
        si_mat = si_mat.at[3:6, 0:3].set(params.m * skew(params.com_pos))


        acc = jnp.linalg.solve(
            si_mat, jnp.concatenate([total_force, total_torque])
        )
        
        # Semi-implicit Euler integration
        vel = state.vel + params.dt * orientation_mat @ acc[:3]
        omega = state.omega + params.dt * acc[3:6]

        pos = state.pos + params.dt * vel
        omega_quat = jnp.array([0.0, omega[0], omega[1], omega[2]])
        q = state.attitude + params.dt * 0.5 * quat_product(quat, omega_quat)

        q = q / jnp.linalg.norm(q)

        # clip position
        # pos = jnp.clip(pos, params.env_min_limits, params.env_max_limits)

        state_new = ModelState(
            pos=pos.squeeze(),
            attitude=q.squeeze(),
            vel=vel.squeeze(),
            omega=omega.squeeze(),
        )

        return state_new

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        pos = jax.random.uniform(
            key,
            shape=(3,),
            minval=params.env_min_limits,
            maxval=params.env_max_limits,
        )
        attitude = jnp.array([1.0, 0.0, 0.0, 0.0])
        vel = jnp.array([0.0, 0.0, 0.0])
        omega = jnp.array([0.0, 0.0, 0.0])
        state = ModelState(pos=pos, attitude=attitude, vel=vel, omega=omega)

        key, key_target, key_yaw = jax.random.split(key, 3)
        target_pos = jax.lax.select(
            params.is_testing,
            self.target_pos,
            jax.random.uniform(
                key_target,
                shape=(3,),
                minval=params.env_min_limits,
                maxval=params.env_max_limits,
            ),
        )
        
        # Sample target yaw uniformly between -pi and pi
        target_yaw = jax.lax.select(
            params.is_testing,
            jnp.array(0.5),  # Point forward when testing
            jax.random.uniform(key_yaw, minval=-jnp.pi, maxval=jnp.pi),
        )

        env_state = EnvState(
            state=state,
            target_pos=target_pos,
            target_yaw=target_yaw,
            last_u=self.hovering_input,
            time=0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, env_state: EnvState) -> jax.Array:
        """Return a vector of observations from the state."""
        state = env_state.state

        pos = state.pos
        vel = state.vel
        to_target = env_state.target_pos - pos
        q = state.attitude
        ang_vel = state.omega
        
        # Compute current yaw and yaw error
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                jnp.cos(current_yaw - env_state.target_yaw))

        return jnp.concatenate(
            [
                to_target.reshape(-1),
                vel.reshape(-1),
                q.reshape(-1),
                ang_vel.reshape(-1),
                jnp.array([yaw_error]),
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)
        
    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Return a discount of 1.0 since the only termination is time limit (truncation)."""
        return jnp.array(1.0, dtype=jnp.float32)

    def eval_metrics(self, env_state: EnvState, params: EnvParams) -> dict[str, jax.Array]:
        """Compute evaluation metrics for the current state.
        
        Args:
            env_state: Current environment state
            params: Environment parameters
            
        Returns:
            Dictionary of metric names to scalar values
        """
        state = env_state.state
        
        # Distance to target
        distance_to_target = jnp.linalg.norm(state.pos - env_state.target_pos)
        
        # Velocity magnitude
        velocity_magnitude = jnp.linalg.norm(state.vel)
        
        # Angular velocity magnitude
        angular_velocity_magnitude = jnp.linalg.norm(state.omega)
        
        # Attitude error (deviation from upright quaternion [1, 0, 0, 0])
        upright_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        quat_error = quat_product(quat_inverse(state.attitude), upright_quat)
        attitude_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(quat_error[0]), 0.0, 1.0))
        
        # Yaw error
        q = state.attitude
        current_yaw = jnp.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2]**2 + q[3]**2))
        yaw_error = jnp.abs(jnp.arctan2(jnp.sin(current_yaw - env_state.target_yaw), 
                                         jnp.cos(current_yaw - env_state.target_yaw)))
        
        return {
            "distance_to_target": distance_to_target,
            "velocity_magnitude": velocity_magnitude,
            "angular_velocity_magnitude": angular_velocity_magnitude,
            "attitude_error": attitude_error,
            "yaw_error": yaw_error,
            "current_yaw": current_yaw,
            "target_yaw": env_state.target_yaw,
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "Drone-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(4,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-1, 1, shape=(self.obs_dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "state": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (13,),
                    jnp.float32,
                ),
                "target_pos": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (3,),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def render_frame_from_state(
        self,
        env_state: EnvState | ModelState,
        trail: list[np.ndarray],
        ax: Any,
    ) -> np.ndarray:
        import matplotlib.pyplot as plt

        state = env_state.state if hasattr(env_state, "state") else env_state
        
        pos = np.asarray(jax.device_get(state.pos))
        attitude = np.asarray(jax.device_get(state.attitude))
        rot = np.asarray(jax.device_get(quat2rotm(attitude)))
        
        target_pos = None
        if hasattr(env_state, "target_pos"):
            target_pos = np.asarray(jax.device_get(env_state.target_pos))
        
        ax.clear()

        # Static scale
        params = self.default_params
        ax.set_xlim([float(params.env_min_limits[0]), float(params.env_max_limits[0])])
        ax.set_ylim([float(params.env_min_limits[1]), float(params.env_max_limits[1])])
        ax.set_zlim([float(params.env_min_limits[2]), float(params.env_max_limits[2])])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # drone as a 3D cross
        l = 0.1  # arm length
        
        # Arms in body frame
        x_axis = np.array([l, 0, 0])
        y_axis = np.array([0, l, 0])
        z_axis = np.array([0, 0, l])
        
        arm1 = np.array([l*np.cos(np.pi/4), l*np.sin(np.pi/4), 0])
        arm2 = np.array([-l*np.cos(np.pi/4), -l*np.sin(np.pi/4), 0])
        arm3 = np.array([l*np.cos(np.pi/4), -l*np.sin(np.pi/4), 0])
        arm4 = np.array([-l*np.cos(np.pi/4), l*np.sin(np.pi/4), 0])
        
        # End points in world frame
        p1 = pos + rot @ arm1
        p2 = pos + rot @ arm2
        p3 = pos + rot @ arm3
        p4 = pos + rot @ arm4
        
        # Draw arms
        ax.plot([pos[0], p1[0]], [pos[1], p1[1]], [pos[2], p1[2]], color='k', linewidth=2)
        ax.plot([pos[0], p2[0]], [pos[1], p2[1]], [pos[2], p2[2]], color='k', linewidth=2)
        ax.plot([pos[0], p3[0]], [pos[1], p3[1]], [pos[2], p3[2]], color='k', linewidth=2)
        ax.plot([pos[0], p4[0]], [pos[1], p4[1]], [pos[2], p4[2]], color='k', linewidth=2)
        
        # Vertical lines at extremes (z axis of body frame)
        h = 0.05
        up = rot @ np.array([0, 0, h])
        for p in [p1, p2, p3, p4]:
            top = p + up / 2
            bot = p - up / 2
            ax.plot([bot[0], top[0]], [bot[1], top[1]], [bot[2], top[2]], color='blue', linewidth=2)
            
        # Heading (along x axis body frame)
        heading_end = pos + rot @ (x_axis * 1.5)
        ax.plot([pos[0], heading_end[0]], [pos[1], heading_end[1]], [pos[2], heading_end[2]], color='r', linewidth=2)
        
        # Trail
        if len(trail) > 0:
            trail_arr = np.array(trail)
            ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], color='gray', alpha=0.5, linestyle='--')
            
        # Target
        if target_pos is not None:
            ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='green', marker='x', s=100, label='Target')
            
        # Convert figure to array
        fig = ax.figure
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        return img

    def render_video_from_states(
        self,
        states: list[EnvState | ModelState],
        output_path: Path,
        fps: int | None = None,
        width: int = 640,
        height: int = 480,
    ) -> Path | None:
        """Render a sequence of states to an MP4 video file using matplotlib."""
        import matplotlib.pyplot as plt
        import imageio

        if fps is None:
            fps = int(1.0 / float(self.default_params.dt))

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        
        frames = []
        trail = []
        try:
            for state_obj in states:
                state = state_obj.state if hasattr(state_obj, "state") else state_obj
                trail.append(np.asarray(jax.device_get(state.pos)))
                frame = self.render_frame_from_state(state_obj, trail, ax)
                frames.append(frame)
        except Exception as exc:
            print(f"[render] Skipping video render due to renderer error: {exc}")
            plt.close(fig)
            return None

        plt.close(fig)

        if not frames:
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps)
        return output_path

    def render(self, env_state: EnvState | ModelState):
        """Render a single frame from the provided state using matplotlib."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        
        trail = []
        state = env_state.state if hasattr(env_state, "state") else env_state
        trail.append(np.asarray(jax.device_get(state.pos)))
        
        img = self.render_frame_from_state(env_state, trail, ax)
        plt.close(fig)
        return img

    
    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot quadrotor rollout trajectories.
        
        Args:
            series_list: List of trajectory dictionaries with keys:
                - 'state': array of shape (T+1, 13) with [pos(3), attitude(4), vel(3), omega(3)]
                - 'actions': array of shape (T, 4)
                - 'target_pos': optional target position array
                - 'eval_metrics': optional dict with 'distance_to_target' array
            output: Optional path to save the figure. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], width_ratios=[1, 1, 1, 1])

        ax_path = fig.add_subplot(gs[0, :2], projection="3d")
        ax_dist = fig.add_subplot(gs[0, 2:])
        axes_act = [fig.add_subplot(gs[1, i]) for i in range(4)]

        labels = ["Thrust", "Torque X", "Torque Y", "Torque Z"]

        # Collect distance_to_target and yaw_error from all rollouts
        distance_series = []
        yaw_series = []
        default_dt = 0.02

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            state = np.asarray(state)
            # state format: [pos(3), attitude(4), vel(3), omega(3)]
            pos = state[:, :3]
            
            ax_path.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.6)

            if "target_pos" in series:
                target = np.asarray(series["target_pos"])
                if target.ndim >= 2:
                    target = target[0]
                ax_path.scatter(
                    target[0], target[1], target[2], marker="x", color="red", s=30
                )

            ax_path.scatter(
                pos[0, 0], pos[0, 1], pos[0, 2], marker="o", color="green", s=30
            )

            actions = np.asarray(series.get("actions", []))
            if actions.size > 0:
                for idx, ax in enumerate(axes_act):
                    if idx < actions.shape[1]:
                        ax.plot(np.arange(actions.shape[0]), actions[:, idx], alpha=0.6)
                    ax.set_xlabel("t")
                    ax.set_ylabel(labels[idx])
                    ax.grid(True, alpha=0.3)

            # Collect distance_to_target and yaw_error if available
            eval_metrics = series.get("eval_metrics", {})
            series_dt = float(series.get("dt", default_dt))
            if "distance_to_target" in eval_metrics:
                dist = np.asarray(eval_metrics["distance_to_target"]).flatten()
                distance_series.append({"values": dist, "dt": series_dt})
            if "yaw_error" in eval_metrics:
                yaw = np.asarray(eval_metrics["yaw_error"]).flatten()
                yaw_series.append({"values": yaw, "dt": series_dt})

        ax_path.set_title("Quadrotor position trajectories")
        ax_path.set_xlabel("x [m]")
        ax_path.set_ylabel("y [m]")
        ax_path.set_zlabel("z [m]")
        ax_path.grid(True, alpha=0.3)

        # Plot mean distance to target and yaw error with std shading
        if distance_series or yaw_series:
            # Create twin axis for yaw error
            ax_yaw = ax_dist.twinx()
            
            # Plot distance to target
            if distance_series:
                max_len = max(d["values"].shape[0] for d in distance_series)
                padded = np.array([
                    np.pad(d["values"], (0, max_len - d["values"].shape[0]), mode='edge') 
                    for d in distance_series
                ])
                mean_dist = np.mean(padded, axis=0)
                std_dist = np.std(padded, axis=0)
                reference_dt = distance_series[0]["dt"] if distance_series else default_dt
                timesteps = np.arange(max_len) * reference_dt

                for dist_entry in distance_series:
                    ax_dist.plot(
                        np.arange(dist_entry["values"].shape[0]) * dist_entry["dt"],
                        dist_entry["values"],
                        color="blue",
                        alpha=0.2,
                        linewidth=0.8,
                        zorder=1,
                    )

                ax_dist.plot(
                    timesteps,
                    mean_dist,
                    color="blue",
                    label="Distance",
                    linewidth=2,
                    zorder=2,
                )
                ax_dist.fill_between(
                    timesteps,
                    mean_dist - std_dist,
                    mean_dist + std_dist,
                    alpha=0.3,
                    color="blue"
                )
                ax_dist.set_ylabel("Distance to target [m]", color="blue")
                ax_dist.tick_params(axis='y', labelcolor='blue')
                
                # Display final mean and std
                final_mean = mean_dist[-1]
                final_std = std_dist[-1]
                ax_dist.text(
                    0.02, 0.98, 
                    f"Distance: {final_mean:.3f} ± {final_std:.3f} m",
                    transform=ax_dist.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
                )
            
            # Plot yaw error
            if yaw_series:
                max_len_yaw = max(y["values"].shape[0] for y in yaw_series)
                padded_yaw = np.array([
                    np.pad(y["values"], (0, max_len_yaw - y["values"].shape[0]), mode='edge') 
                    for y in yaw_series
                ])
                mean_yaw = np.mean(padded_yaw, axis=0)
                std_yaw = np.std(padded_yaw, axis=0)
                reference_dt = distance_series[0]["dt"] if distance_series else yaw_series[0]["dt"]
                timesteps_yaw = np.arange(max_len_yaw) * reference_dt

                for yaw_entry in yaw_series:
                    ax_yaw.plot(
                        np.arange(yaw_entry["values"].shape[0]) * yaw_entry["dt"],
                        yaw_entry["values"],
                        color="red",
                        alpha=0.2,
                        linewidth=0.8,
                        zorder=1,
                    )

                ax_yaw.plot(timesteps_yaw, mean_yaw, color="red", label="Yaw error", linewidth=2)
                ax_yaw.fill_between(
                    timesteps_yaw,
                    mean_yaw - std_yaw,
                    mean_yaw + std_yaw,
                    alpha=0.3,
                    color="red"
                )
                ax_yaw.set_ylabel("Yaw error [rad]", color="red")
                ax_yaw.tick_params(axis='y', labelcolor='red')
                
                # Display final yaw mean and std
                final_yaw_mean = mean_yaw[-1]
                final_yaw_std = std_yaw[-1]
                ax_dist.text(
                    0.02, 0.88, 
                    f"Yaw error: {final_yaw_mean:.3f} ± {final_yaw_std:.3f} rad",
                    transform=ax_dist.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5)
                )
            
            ax_dist.set_title("Distance and Yaw Error over time")
            ax_dist.set_xlabel("Time [s]")
            ax_dist.grid(True, alpha=0.3)
            dist_top = max(ax_dist.get_ylim()[1], 0)
            ax_dist.set_ylim(0, dist_top)
            yaw_top = max(ax_yaw.get_ylim()[1], 0)
            ax_yaw.set_ylim(0, yaw_top)
            
            # Add legend
            lines1, labels1 = ax_dist.get_legend_handles_labels()
            lines2, labels2 = ax_yaw.get_legend_handles_labels()
            ax_dist.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax_dist.text(0.5, 0.5, "No eval_metrics available", 
                        ha='center', va='center', transform=ax_dist.transAxes)
            ax_dist.set_title("Distance and Yaw Error over time")

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)
