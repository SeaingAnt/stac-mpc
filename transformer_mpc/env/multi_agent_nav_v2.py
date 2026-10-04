"""Multi-Agent 2D Navigation Environment.

N robots navigate a 3×3 m arena to reach individual target positions
while avoiding collisions with each other.  Each robot is a disc of
radius 0.15 m governed by single-integrator dynamics (velocity control).

Observation design (CTDE – Centralised Training, Decentralised Execution):
  • local_obs  – per-agent: relative vector to own target + pairwise
                 distances/bearings to every other robot.
  • global_state – concatenation of all positions + all targets (for the
                   centralized critic).

Reward shaping (positive only):
  • Progress reward: improvement in distance to target (always >= 0).
  • Goal reward: bonus when within goal_radius of target.
  • Alive bonus: small constant each step for surviving.

Termination:
  • Collision between any two robots (distance < 2 * robot_radius).
  • Truncation after max_steps_in_episode.
"""

from functools import partial
from typing import Optional, Tuple, Union, Any, List
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces


# ---------------------------------------------------------------------------
# State & Params dataclasses
# ---------------------------------------------------------------------------

@struct.dataclass
class NavState(environment.EnvState):
    """Full environment state for the multi-agent navigation task."""
    pos: jax.Array        # (num_agents, 2) – positions in metres
    target: jax.Array     # (num_agents, 2) – target positions
    prev_dist: jax.Array  # (num_agents,)   – distance to target at prev step
    global_state: jax.Array  # Full state vector for the centralized critic
    time: int
    history: jax.Array       # (num_agents, seq_len, local_obs_dim)


@struct.dataclass
class NavParams(environment.EnvParams):
    """Environment parameters (all lengths in metres, time in seconds)."""
    num_agents: int = struct.field(pytree_node=False, default=4)
    arena_size: float = struct.field(pytree_node=False, default=3.0)
    robot_radius: float = struct.field(pytree_node=False, default=0.15)
    max_speed: float = struct.field(pytree_node=False, default=0.5)
    dt: float = struct.field(pytree_node=False, default=0.1)
    goal_radius: float = struct.field(pytree_node=False, default=0.1)
    goal_bonus: float = struct.field(pytree_node=False, default=10.0)
    alive_bonus: float = struct.field(pytree_node=False, default=0.1)
    progress_scale: float = struct.field(pytree_node=False, default=5.0)
    max_steps_in_episode: int = struct.field(pytree_node=False, default=100)
    mpc_horizon: int = struct.field(pytree_node=False, default=10)
    lidar_bins: int = struct.field(pytree_node=False, default=10)
    lidar_range: float = struct.field(pytree_node=False, default=0.40)
    # Action bounds (normalised)
    min_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([-1.0, -1.0]),
    )
    max_action: jax.Array = struct.field(
        pytree_node=False,
        default_factory=lambda: jnp.array([1.0, 1.0]),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MultiAgentNav2DV2(environment.Environment):
    """JAX-compatible multi-agent 2D navigation with history buffer for sequences."""

    # Flag used by MAPPO's wrap_env to ensure the env is multi-agent.
    decentralized = True

    def __init__(self, num_agents: int = 4, **kwargs):
        self._num_agents = num_agents
        super().__init__()

    # ----- properties -----

    @property
    def default_params(self) -> NavParams:
        return NavParams(num_agents=self._num_agents)

    class mpc_dynamics:
        def __init__(self, state_dim, num_actions, env_params):
            self.max_speed = env_params.max_speed

        def state_dot(self, state, control, params):
            # state is (N, 2), control is (N, 2)
            # but mpx expects 1D arrays, so state is (2*N,), control is (2*N,)
            return self.max_speed * control

    @property
    def num_agents(self) -> int:
        return self._num_agents

    # ----- helpers -----

    @staticmethod
    def _pairwise_dists(pos):
        """(N, 2) -> (N, N) pairwise Euclidean distances."""
        diff = pos[:, None, :] - pos[None, :, :]        # (N, N, 2)
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # (N, N)

    @staticmethod
    def _check_collision(pos, robot_radius):
        """Return True (scalar) if any two robots overlap."""
        dists = MultiAgentNav2DV2._pairwise_dists(pos)        # (N, N)
        n = pos.shape[0]
        # Mask out diagonal (self-distances)
        mask = ~jnp.eye(n, dtype=bool)
        collision_threshold = 2.0 * robot_radius
        return jnp.any((dists < collision_threshold) & mask)

    @staticmethod
    def _compute_lidar(pos: jax.Array, params: NavParams) -> jax.Array:
        # pos: (N, 2)
        n = params.num_agents
        k = params.lidar_bins
        angles = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)
        dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (K, 2)

        v = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
        v_dot_d = jnp.einsum('nij,kj->nik', v, dirs)  # (N, N, K)
        
        v_sq = jnp.sum(v ** 2, axis=-1, keepdims=True)  # (N, N, 1)
        r_sq = params.robot_radius ** 2
        
        delta = v_dot_d ** 2 - (v_sq - r_sq)  # (N, N, K)
        
        valid_delta = delta >= 0
        t1 = -v_dot_d - jnp.sqrt(jnp.maximum(0.0, delta))
        t2 = -v_dot_d + jnp.sqrt(jnp.maximum(0.0, delta))
        
        t = jnp.where(t1 > 0, t1, t2)
        valid_t = t > 0
        
        valid = valid_delta & valid_t & (~jnp.eye(n, dtype=bool))[:, :, None]
        
        dists = jnp.where(valid, t, params.lidar_range)
        lidar = jnp.min(dists, axis=1)  # (N, K)
        return jnp.clip(lidar, 0.0, params.lidar_range)

    def _single_obs(self, state: NavState, params: NavParams):
        """Build the single-step local_obs for all agents."""
        n = params.num_agents
        pos = state.pos       # (N, 2)
        target = state.target  # (N, 2)

        rel_target = target - pos  # (N, 2)
        dist_target = jnp.linalg.norm(rel_target, axis=-1, keepdims=True)  # (N, 1)
        norm_pos = pos / (params.arena_size / 2.0) - 1.0  # (N, 2) in [-1, 1]

        lidar = self._compute_lidar(pos, params)  # (N, K)

        own_features = jnp.concatenate([rel_target, dist_target], axis=-1)  # (N, 3)

        local_obs = jnp.concatenate([own_features, norm_pos, lidar], axis=-1)  # (N, D_local)
        return local_obs

    def _make_obs(self, state: NavState, params: NavParams):
        """Build the dict observation: local_obs (history) and global_state."""
        local_obs = state.history # (N, seq_len, D_local)
        current_local_obs = state.history[:, -1, :] # (N, D_local)

        time_frac = jnp.array([state.time / params.max_steps_in_episode])
        global_state = jnp.concatenate([
            current_local_obs.reshape(-1),
            time_frac,
        ])  

        return {"local_obs": local_obs, "global_state": global_state}

    # ----- core API -----

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.Array,
        params: Optional[NavParams] = None,
    ) -> Tuple[dict, NavState]:
        if params is None:
            params = self.default_params

        n = params.num_agents
        margin = params.robot_radius * 2.0
        lo = margin
        hi = params.arena_size - margin

        # Sample positions ensuring no initial overlaps via rejection-free approach:
        # Place agents on a larger grid and randomly choose subsets
        grid_size = 4  # 16 possible spawn points
        xs = jnp.linspace(lo, hi, grid_size)
        ys = jnp.linspace(lo, hi, grid_size)
        xx, yy = jnp.meshgrid(xs, ys)
        all_grid_pos = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)
        
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        pos_indices = jax.random.choice(k1, grid_size**2, shape=(n,), replace=False)
        pos = all_grid_pos[pos_indices]
        
        target_indices = jax.random.choice(k2, grid_size**2, shape=(n,), replace=False)
        target = all_grid_pos[target_indices]
        
        # Add noise to make it continuous and more randomized
        noise_pos = jax.random.uniform(k3, (n, 2), minval=-0.2, maxval=0.2)
        noise_target = jax.random.uniform(k4, (n, 2), minval=-0.2, maxval=0.2)
        
        pos = jnp.clip(pos + noise_pos, lo, hi)
        target = jnp.clip(target + noise_target, lo, hi)

        dist_to_target = jnp.linalg.norm(target - pos, axis=-1)

        temp_state = NavState(
            pos=pos,
            target=target,
            prev_dist=dist_to_target,
            global_state=jnp.zeros(0),
            time=0,
            history=jnp.zeros((n, params.mpc_horizon + 1, 5 + params.lidar_bins)), # placeholder
        )
        single_obs = self._single_obs(temp_state, params) # (num_agents, local_obs_dim)
        
        seq_len = params.mpc_horizon + 1
        history = jnp.tile(single_obs[:, None, :], (1, seq_len, 1))
        
        temp_state = temp_state.replace(history=history)

        obs = self._make_obs(temp_state, params)
        state = temp_state.replace(global_state=obs["global_state"])
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: NavState,
        action: jax.Array,
        params: Optional[NavParams] = None,
    ) -> Tuple[dict, NavState, float, bool, dict]:
        if params is None:
            params = self.default_params

        n = params.num_agents

        # action: (num_agents, 2) normalised in [-1, 1]
        # If flattened by wrappers, reshape
        act = action.reshape(n, 2)
        act = jnp.clip(act, -1.0, 1.0)

        # Single integrator: velocity = max_speed * action
        velocity = params.max_speed * act  # (N, 2)

        # Euler integration
        new_pos = state.pos + params.dt * velocity

        # Clamp to arena boundaries
        new_pos = jnp.clip(new_pos, params.robot_radius, params.arena_size - params.robot_radius)

        # --- Collision check ---
        collision = self._check_collision(new_pos, params.robot_radius)

        # --- Reward (positive only) ---
        dist_to_target = jnp.linalg.norm(state.target - new_pos, axis=-1)  # (N,)

        # Closeness reward: strictly positive, increases as agents get closer to target
        # Maximize exp(-dist) encourages dist -> 0
        closeness_reward = params.progress_scale * jnp.exp(-dist_to_target).sum()

        # Goal reward: bonus for each agent within goal_radius
        at_goal = dist_to_target < params.goal_radius  # (N,)
        goal_reward = params.goal_bonus * at_goal.sum()

        # Alive bonus
        alive_reward = params.alive_bonus

        # Total reward is shared across all agents (team reward)
        reward = closeness_reward + goal_reward + alive_reward

        # --- Done ---
        time_up = (state.time + 1) >= params.max_steps_in_episode
        done = collision | time_up

        # --- New state ---
        temp_new_state = NavState(
            pos=new_pos,
            target=state.target,
            prev_dist=dist_to_target,
            global_state=jnp.zeros(0),
            time=state.time + 1,
            history=state.history,
        )
        
        single_obs = self._single_obs(temp_new_state, params) # (num_agents, local_obs_dim)
        
        # update history: (num_agents, seq_len, obs_dim)
        new_history = jnp.concatenate([state.history[:, 1:, :], single_obs[:, None, :]], axis=1)
        temp_new_state = temp_new_state.replace(history=new_history)

        obs = self._make_obs(temp_new_state, params)
        new_state = temp_new_state.replace(global_state=obs["global_state"])

        info = {
            "collision": collision,
            "at_goal": at_goal.sum(),
            "mean_dist_to_target": dist_to_target.mean(),
            # Provide discount for truncation handling in GAE
            "discount": jnp.where(collision, 0.0, 1.0),
        }

        return obs, new_state, reward, done, info

    # ----- spaces -----

    def action_space(self, params: Optional[NavParams] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(params.num_agents, 2),
            dtype=jnp.float32,
        )

    def observation_space(self, params: Optional[NavParams] = None) -> spaces.Dict:
        if params is None:
            params = self.default_params
        n = params.num_agents
        # local_obs: 3 (rel target) + 2 (norm pos) + lidar_bins
        local_obs_dim = 5 + params.lidar_bins
        seq_len = params.mpc_horizon + 1
        # global_state: flatten(current_local_obs) + 1 (time frac)
        global_state_dim = n * local_obs_dim + 1
        return spaces.Dict({
            "local_obs": spaces.Box(
                low=-jnp.inf, high=jnp.inf,
                shape=(n, seq_len, local_obs_dim),
                dtype=jnp.float32,
            ),
            "global_state": spaces.Box(
                low=-jnp.inf, high=jnp.inf,
                shape=(global_state_dim,),
                dtype=jnp.float32,
            ),
        })

    def state_space(self, params: Optional[NavParams] = None) -> spaces.Dict:
        if params is None:
            params = self.default_params
        n = params.num_agents
        return spaces.Dict({
            "pos": spaces.Box(
                low=0.0, high=params.arena_size,
                shape=(n, 2), dtype=jnp.float32,
            ),
            "target": spaces.Box(
                low=0.0, high=params.arena_size,
                shape=(n, 2), dtype=jnp.float32,
            ),
        })

    def render_frame_from_state(
        self,
        env_state: NavState | environment.EnvState,
        trail: list[np.ndarray],
        ax: Any,
    ) -> np.ndarray:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        state = env_state.state if hasattr(env_state, "state") else env_state
        
        pos = np.asarray(jax.device_get(state.pos))
        target = np.asarray(jax.device_get(state.target))
        
        ax.clear()

        # Static scale
        params = self.default_params
        ax.set_xlim([0.0, params.arena_size])
        ax.set_ylim([0.0, params.arena_size])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, self.num_agents))
        
        for i in range(self.num_agents):
            c = colors[i % len(colors)]
            
            # Draw robot
            circle = patches.Circle((pos[i, 0], pos[i, 1]), params.robot_radius, edgecolor=c, facecolor=c, alpha=0.6)
            ax.add_patch(circle)
            
            # Draw target
            ax.scatter(target[i, 0], target[i, 1], color=c, marker='x', s=100, label=f'Target {i}')
            
            # Draw lidar
            angles = np.linspace(0, 2 * np.pi, params.lidar_bins, endpoint=False)
            dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
            
            for k in range(params.lidar_bins):
                min_t = params.lidar_range
                d = dirs[k]
                for j in range(self.num_agents):
                    if i == j: continue
                    v = pos[i] - pos[j]
                    v_dot_d = np.dot(v, d)
                    v_sq = np.dot(v, v)
                    r_sq = params.robot_radius**2
                    delta = v_dot_d**2 - (v_sq - r_sq)
                    if delta >= 0:
                        t1 = -v_dot_d - np.sqrt(delta)
                        t2 = -v_dot_d + np.sqrt(delta)
                        t = t1 if t1 > 0 else t2
                        if t > 0 and t < min_t:
                            min_t = t
                
                ray_end = pos[i] + d * min_t
                ax.plot([pos[i, 0], ray_end[0]], [pos[i, 1], ray_end[1]], color=c, alpha=0.3, linewidth=1)
                ax.scatter(ray_end[0], ray_end[1], color='red', s=5, alpha=0.5)

            # Draw trail
            if len(trail) > 0:
                trail_arr = np.array(trail) # (T, N, 2)
                ax.plot(trail_arr[:, i, 0], trail_arr[:, i, 1], color=c, alpha=0.5, linestyle='--')
            
        # Convert figure to array
        fig = ax.figure
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        return img

    def render_video_from_states(
        self,
        states: list[NavState | environment.EnvState],
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

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        
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

    def render(self, env_state: NavState | environment.EnvState):
        """Render a single frame from the provided state using matplotlib."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        
        trail = []
        state = env_state.state if hasattr(env_state, "state") else env_state
        trail.append(np.asarray(jax.device_get(state.pos)))
        
        img = self.render_frame_from_state(env_state, trail, ax)
        plt.close(fig)
        return img
        
    @staticmethod
    def plot_rollouts(series_list: List[dict], output: Path | None = None) -> None:
        """Plot multi-agent rollout trajectories."""
        import matplotlib.pyplot as plt
        
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = plt.cm.get_cmap('tab10').colors

        for series in series_list:
            state = series.get("state", None)
            if state is None:
                continue
            
            pos = None
            target = None
            if hasattr(state, "pos"):
                pos = np.asarray(state.pos)
                target = np.asarray(state.target)
            elif isinstance(state, np.ndarray):
                if state.ndim == 3 and state.shape[-1] == 2:
                    pos = state
            
            if pos is not None:
                T, num_agents, _ = pos.shape
                for i in range(num_agents):
                    c = colors[i % len(colors)]
                    ax.plot(pos[:, i, 0], pos[:, i, 1], color=c, alpha=0.6, label=f"Agent {i}" if series is series_list[0] else "")
                    ax.scatter(pos[0, i, 0], pos[0, i, 1], marker="o", color=c, s=30)
                    if target is not None:
                        ax.scatter(target[0, i, 0], target[0, i, 1], marker="x", color=c, s=100)

        ax.set_title("Multi-Agent Position Trajectories")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # handle duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()

        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            plt.close(fig)
