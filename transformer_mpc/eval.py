import argparse
import json
import sys
import os
from pathlib import Path

# Force CPU execution for evaluation as XLA GPU compilation of the
# unrolled MPC solver causes "Contracting dimension is too fragmented"
os.environ["JAX_PLATFORMS"] = "cpu"

# Allow XLA to autotune to avoid "Contracting dimension is too fragmented" error
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=3 --xla_gpu_force_compilation_parallelism=16"
)

# Add the project root to sys.path so we can import mpx, etc.
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "mpx"))

import jax
import jax.numpy as jnp
from flax import nnx, serialization

from algorithms import ALGORITHM_REGISTRY
from env import ENV_REGISTRY
from env.math import quat2rotm
from algorithms.base import _policy_dist_from_model
from env.wrappers import EvalVideoWrapper
from buffers import RunnerState

STRONG_COLORS = [
    "#d62728",
    "#2ca02c",
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _build_deterministic_eval_step(config, graphdef, rngs_state, env, env_params):
    """Create deterministic env-step function for evaluation rollouts."""

    def get_physical_state(state):
        def _get_base_state(s):
            if hasattr(s, "pos"):
                return s
            if hasattr(s, "state") and hasattr(s.state, "pos"):
                return s.state
            if hasattr(s, "env_state"):
                return _get_base_state(s.env_state)
            return None

        base = _get_base_state(state)
        if base is None:
            return None
        return jnp.concatenate([base.pos, base.attitude, base.vel, base.omega], axis=-1)

    def _eval_step(model, eval_state: RunnerState, _unused):
        rng, model_rng = jax.random.split(eval_state.rng)
        
        if "DIFFMPC" in config.get("ALGORITHM", "").upper():
            physical_state = get_physical_state(eval_state.env_state)
            
            obs = eval_state.last_obs
            if isinstance(obs, dict) and "local_obs" in obs:
                obs = obs["local_obs"]
                
            model_out = model(obs, physical_state=physical_state)
            pi = model_out[0] if isinstance(model_out, tuple) else model_out
        else:
            pi = _policy_dist_from_model(model, eval_state.last_obs, model_rng)
            
        action = pi.mode()

        rng, step_rng = jax.random.split(rng)
        rng_step = jax.random.split(step_rng, config["NUM_ENVS"])
        obsv, env_state, _reward, _done, info = env.step(
            rng_step, eval_state.env_state, action, env_params
        )

        new_eval_state = RunnerState(
            train_state=eval_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            extras=eval_state.extras,
        )
        return new_eval_state, (info, env_state)

    return _eval_step


def _save_combined_multi_agent_error_trajectory_plot(runs_data, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, item in enumerate(runs_data):
        if len(item) == 3:
            env_state_batch, info_batch, label = item
        else:
            env_state_batch, label = item
            info_batch = {}

        env_state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
        info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)

        while hasattr(env_state_batch, "env_state"):
            env_state_batch = env_state_batch.env_state

        model_state = getattr(env_state_batch, "state", env_state_batch)
        positions = np.asarray(model_state.pos)
        target_pos = np.asarray(env_state_batch.target)

        returned_episode = (
            np.asarray(info_batch["returned_episode"])
            if "returned_episode" in info_batch
            else None
        )

        if "terminal_pos" in info_batch:
            terminal_positions = np.asarray(info_batch["terminal_pos"])
        else:
            terminal_positions = positions

        collisions = (
            np.asarray(info_batch["collision"])
            if "collision" in info_batch
            else None
        )

        num_steps = positions.shape[0]
        num_envs = positions.shape[1] if positions.ndim > 2 else 1
        num_agents = positions.shape[2] if positions.ndim > 3 else 1

        color = STRONG_COLORS[i % len(STRONG_COLORS)]
        clean_trajectories = []

        for env_idx in range(num_envs):
            if returned_episode is not None:
                env_returns = (
                    returned_episode[:, env_idx]
                    if returned_episode.ndim > 1
                    else returned_episode
                )
                done_steps = np.where(env_returns)[0].tolist()
            else:
                done_steps = []

            if len(done_steps) > 0:
                first_done_idx = min(done_steps[0], len(positions) - 1)
                env_pos = np.zeros(
                    (first_done_idx + 1, num_agents, positions.shape[-1]),
                    dtype=positions.dtype,
                )
                env_pos[:first_done_idx] = positions[:first_done_idx, env_idx]
                env_pos[first_done_idx] = terminal_positions[first_done_idx, env_idx]
                has_collided = (
                    bool(np.any(collisions[: first_done_idx + 1, env_idx]))
                    if collisions is not None
                    else False
                )
            else:
                first_done_idx = len(positions) - 1
                env_pos = positions[:, env_idx]
                has_collided = (
                    bool(np.any(collisions[:, env_idx]))
                    if collisions is not None
                    else False
                )

            # Only retain trajectories that did not collide
            if has_collided:
                continue

            env_target = (
                target_pos[0, env_idx]
                if target_pos.ndim == 4
                else target_pos[env_idx]
            )

            # Euclidean error of each agent with respect to its target at each step
            diff = env_pos - env_target[None, :, :]
            agent_dist = np.linalg.norm(diff, axis=-1)  # shape: (T, num_agents)
            team_dist = agent_dist.mean(axis=-1)        # shape: (T,)

            clean_trajectories.append(team_dist)

        if len(clean_trajectories) == 0:
            print(f"Notice for plot: Policy '{label}' has 0 non-colliding trajectories.")
            ax.plot([], [], color=color, linestyle="--", label=f"{label} (0/{num_envs} clean)")
            continue

        # Plot individual non-colliding trajectories
        for traj in clean_trajectories:
            ax.plot(
                np.arange(len(traj)),
                traj,
                color=color,
                alpha=0.15,
                linewidth=0.9,
            )

        # Compute mean and standard deviation across non-colliding trajectories
        max_len = max(len(t) for t in clean_trajectories)
        time_axis = np.arange(max_len)
        mean_curve = np.zeros(max_len)
        std_curve = np.zeros(max_len)

        for t in range(max_len):
            vals = [traj[t] for traj in clean_trajectories if t < len(traj)]
            mean_curve[t] = np.mean(vals)
            std_curve[t] = np.std(vals) if len(vals) > 1 else 0.0

        clean_label = f"{label} ({len(clean_trajectories)}/{num_envs} clean)"
        ax.plot(time_axis, mean_curve, color=color, linewidth=2.5, label=clean_label)
        ax.fill_between(
            time_axis,
            np.maximum(mean_curve - std_curve, 0.0),
            mean_curve + std_curve,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Distance to target [m]", fontsize=16)
    ax.set_title("Multi-agent error to target over time (non-colliding)", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=14)
    ax.set_ylim(bottom=-0.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved multi-agent error trajectory plot to {output_path}")


def _save_combined_error_trajectory_plot(runs_data, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    sample_state = runs_data[0][0]
    while hasattr(sample_state, "env_state"):
        sample_state = sample_state.env_state
    if hasattr(sample_state, "target"):
        return _save_combined_multi_agent_error_trajectory_plot(runs_data, output_path)

    fig, (ax_pos, ax_yaw) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for i, item in enumerate(runs_data):
        env_state_batch = item[0]
        label = item[-1]
        env_state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)

        while hasattr(env_state_batch, "env_state"):
            env_state_batch = env_state_batch.env_state

        model_state = getattr(env_state_batch, "state", env_state_batch)
        
        has_yaw = False
        if hasattr(env_state_batch, "target"):
            target_pos = np.asarray(env_state_batch.target)
            positions = np.asarray(model_state.pos)
        else:
            target_pos = np.asarray(env_state_batch.target_pos)
            positions = np.asarray(model_state.pos)
            target_yaw = np.asarray(env_state_batch.target_yaw)
            attitudes = np.asarray(model_state.attitude)
            has_yaw = True

        pos_error = np.linalg.norm(positions - target_pos, axis=-1)
        if pos_error.ndim == 3:  # (steps, envs, agents)
            pos_error = pos_error.mean(axis=-1)

        if has_yaw:
            q = attitudes
            current_yaw = np.arctan2(
                2.0 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2),
            )
            yaw_error = np.abs(
                np.arctan2(
                    np.sin(current_yaw - target_yaw), np.cos(current_yaw - target_yaw)
                )
            )
            yaw_mean = yaw_error.mean(axis=1)
            yaw_std = yaw_error.std(axis=1)

        time_axis = np.arange(pos_error.shape[0])
        pos_mean = pos_error.mean(axis=1)
        pos_std = pos_error.std(axis=1)

        color = STRONG_COLORS[i % len(STRONG_COLORS)]

        for episode_idx in range(pos_error.shape[1]):
            ax_pos.plot(
                time_axis,
                pos_error[:, episode_idx],
                color=color,
                alpha=0.1,
                linewidth=0.9,
            )
            if has_yaw:
                ax_yaw.plot(
                    time_axis,
                    yaw_error[:, episode_idx],
                    color=color,
                    alpha=0.1,
                    linewidth=0.9,
                )

        ax_pos.plot(
            time_axis, pos_mean, color=color, linewidth=2, label=f"Distance ({label})"
        )
        ax_pos.fill_between(
            time_axis, jnp.clip(pos_mean - pos_std, 0, 100), pos_mean + pos_std, color=color, alpha=0.2
        )

        ax_pos.set_ylim(-0.2, 5.0) 

        if has_yaw:
            ax_yaw.plot(
                time_axis, yaw_mean, color=color, linewidth=2, label=f"Yaw ({label})"
            )
            ax_yaw.fill_between(
                time_axis, jnp.clip(yaw_mean - yaw_std, 0, 100), yaw_mean + yaw_std, color=color, alpha=0.2
            )

    ax_pos.set_ylabel("Position error [m]", fontsize=16)
    ax_pos.set_title("Tracking error over time", fontsize=18)
    ax_pos.tick_params(axis="both", which="major", labelsize=14)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="upper right", fontsize=14)

    ax_yaw.set_xlabel("Step", fontsize=16)
    ax_yaw.set_ylabel("Yaw error [rad]", fontsize=16)
    ax_yaw.tick_params(axis="both", which="major", labelsize=14)
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.legend(loc="upper right", fontsize=14)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_transparent_quadrotor(ax, pos, attitude, alpha=0.2, color="k"):
    import numpy as np

    rot = np.asarray(jax.device_get(quat2rotm(attitude)))
    pos = np.asarray(pos)

    l = 0.1
    arm1 = np.array([l * np.cos(np.pi / 4), l * np.sin(np.pi / 4), 0.0])
    arm2 = np.array([-l * np.cos(np.pi / 4), -l * np.sin(np.pi / 4), 0.0])
    arm3 = np.array([l * np.cos(np.pi / 4), -l * np.sin(np.pi / 4), 0.0])
    arm4 = np.array([-l * np.cos(np.pi / 4), l * np.sin(np.pi / 4), 0.0])

    p1 = pos + rot @ arm1
    p2 = pos + rot @ arm2
    p3 = pos + rot @ arm3
    p4 = pos + rot @ arm4

    for point in [p1, p2, p3, p4]:
        ax.plot(
            [pos[0], point[0]],
            [pos[1], point[1]],
            [pos[2], point[2]],
            color=color,
            alpha=alpha,
            linewidth=2,
        )

    h = 0.05
    up = rot @ np.array([0, 0, h])
    for p in [p1, p2, p3, p4]:
        top = p + up / 2
        bot = p - up / 2
        ax.plot(
            [bot[0], top[0]],
            [bot[1], top[1]],
            [bot[2], top[2]],
            color="blue",
            alpha=alpha,
            linewidth=2,
        )

    x_axis = np.array([l, 0, 0])
    heading_end = pos + rot @ (x_axis * 1.5)
    ax.plot(
        [pos[0], heading_end[0]],
        [pos[1], heading_end[1]],
        [pos[2], heading_end[2]],
        color="black",  # Keep heading indicator black or strong red, distinct from body
        alpha=alpha,
        linewidth=2,
    )


def _save_combined_collision_rate_bar_plot(runs_data, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    labels = []
    rates = []

    for _, info_batch, label in runs_data:
        info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)
        if "collision" in info_batch:
            collisions = np.asarray(info_batch["collision"])
            returned_episode = np.asarray(info_batch["returned_episode"])
            num_done = np.sum(returned_episode)
            rate = (np.sum(collisions) / num_done * 100) if num_done > 0 else 0.0
        else:
            rate = 0.0
        labels.append(label)
        rates.append(rate)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 6))
    bars = ax.bar(labels, rates, color=STRONG_COLORS[:len(labels)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel("Collision Rate (%)", fontsize=16)
    ax.set_title("Collision Rate per Policy", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(rates) * 1.2, 5.0)])

    # Annotate bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{rate:.1f}%",
            ha='center', va='bottom', fontsize=14
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def _save_combined_trajectory_2d_plot(runs_data, output_path, env_params=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 8))

    arena_size = float(getattr(env_params, "arena_size", 3.0)) if env_params else 3.0
    robot_radius = float(getattr(env_params, "robot_radius", 0.15)) if env_params else 0.15

    ax.set_xlim([0, arena_size])
    ax.set_ylim([0, arena_size])

    ax.set_xlabel("X [m]", fontsize=16)
    ax.set_ylabel("Y [m]", fontsize=16)
    ax.set_title("2D trajectory with ghost motion", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True, linestyle=":", alpha=0.3)

    linestyles = ['-', '--', '-.', ':']
    colors = plt.colormaps['tab10'](np.linspace(0, 1, 10))

    plotted_targets = set()

    for run_idx, (env_state_batch, info_batch, label) in enumerate(runs_data):
        env_state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
        info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)

        while hasattr(env_state_batch, "env_state"):
            env_state_batch = env_state_batch.env_state

        model_state = getattr(env_state_batch, "state", env_state_batch)
        positions = np.asarray(model_state.pos)
        returned_episode = np.asarray(info_batch["returned_episode"])

        if "terminal_pos" in info_batch:
            terminal_positions = np.asarray(info_batch["terminal_pos"])
        else:
            terminal_positions = positions

        target_pos = np.asarray(env_state_batch.target)
        num_agents = positions.shape[2]
        style = linestyles[run_idx % len(linestyles)]

        num_envs = positions.shape[1]
        for env_idx in range(num_envs):
            env_returns = (
                returned_episode[:, env_idx]
                if returned_episode.ndim > 1
                else returned_episode
            )
            done_steps = np.where(env_returns)[0].tolist()
            if len(done_steps) > 0:
                first_done_idx = min(done_steps[0], len(positions) - 1)
                env_pos = np.zeros(
                    (first_done_idx + 1, num_agents, 2), dtype=positions.dtype
                )
                env_pos[:first_done_idx] = positions[:first_done_idx, env_idx]
                env_pos[first_done_idx] = terminal_positions[first_done_idx, env_idx]
                env_target = target_pos[first_done_idx, env_idx]
            else:
                env_pos = positions[:, env_idx]
                env_target = target_pos[-1, env_idx]

            for agent_idx in range(num_agents):
                agent_pos = env_pos[:, agent_idx]
                c = colors[agent_idx % len(colors)]
                
                # Label only for the first agent to avoid legendary clutter, or label the policy
                ax.plot(
                    agent_pos[:, 0],
                    agent_pos[:, 1],
                    color=c,
                    alpha=0.7,
                    linewidth=2.0,
                    linestyle=style,
                    label=f"{label}" if (agent_idx == 0 and env_idx == 0) else "",
                )

                # Draw ghost robot circles along trajectory
                t_steps = len(agent_pos)
                if t_steps > 1:
                    ghost_step = max(1, t_steps // 8)
                    ghost_indices = list(range(0, t_steps - 1, ghost_step))
                    alphas = np.linspace(0.10, 0.45, len(ghost_indices))
                    for g_i, g_idx in enumerate(ghost_indices):
                        ghost_circle = patches.Circle(
                            (agent_pos[g_idx, 0], agent_pos[g_idx, 1]),
                            robot_radius,
                            edgecolor=c,
                            facecolor=c,
                            alpha=float(alphas[g_i]),
                            linewidth=1.0,
                            zorder=2,
                        )
                        ax.add_patch(ghost_circle)

                # Start marker
                ax.scatter(
                    agent_pos[0, 0],
                    agent_pos[0, 1],
                    color=c,
                    marker="o",
                    s=40,
                    facecolors="white",
                    edgecolors=c,
                    linewidths=1.5,
                    zorder=5,
                )

                # Final robot solid disc
                final_circle = patches.Circle(
                    (agent_pos[-1, 0], agent_pos[-1, 1]),
                    robot_radius,
                    edgecolor="black",
                    facecolor=c,
                    alpha=0.9,
                    linewidth=1.5,
                    zorder=5,
                )
                ax.add_patch(final_circle)

                # Plot targets only once per agent
                t_pos = tuple(np.round(env_target[agent_idx], 3))
                if t_pos not in plotted_targets:
                    ax.scatter(
                        t_pos[0],
                        t_pos[1],
                        color=c,
                        marker="x",
                        s=120,
                        linewidths=2.5,
                        zorder=6,
                        label="Target" if len(plotted_targets) == 0 else "",
                    )
                    plotted_targets.add(t_pos)

    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_combined_trajectory_3d_plot(runs_data, output_path, step_every=50, env_params=None):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Remove grey background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_zlabel("Z", fontsize=16)
    ax.set_title("3D trajectory across all evaluation episodes", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    plotted_targets = set()
    plotted_obstacles = set()
    target_label_added = False
    obstacle_label_added = False

    for i, (env_state_batch, info_batch, label) in enumerate(runs_data):
        env_state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
        info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)

        while hasattr(env_state_batch, "env_state"):
            env_state_batch = env_state_batch.env_state

        model_state = env_state_batch.state

        positions = np.asarray(model_state.pos)
        attitudes = np.asarray(model_state.attitude)
        returned_episode = np.asarray(info_batch["returned_episode"])

        time_steps = positions.shape[0]
        num_envs = positions.shape[1] if positions.ndim > 2 else 1

        color = STRONG_COLORS[i % len(STRONG_COLORS)]

        episode_segments = []
        for env_idx in range(num_envs):
            start_idx = 0
            env_returns = (
                returned_episode[:, env_idx]
                if returned_episode.ndim > 1
                else returned_episode
            )

            done_steps = np.where(env_returns)[0].tolist()
            done_steps.append(time_steps)

            for end_idx in done_steps:
                if end_idx <= start_idx:
                    start_idx = end_idx + 1
                    continue

                episode_segments.append((env_idx, start_idx, end_idx))
                start_idx = end_idx + 1

        for segment_idx, (env_idx, start_idx, end_idx) in enumerate(episode_segments):
            segment_positions = positions[start_idx:end_idx, env_idx]
            segment_attitudes = attitudes[start_idx:end_idx, env_idx]

            if segment_positions.shape[0] == 0:
                continue

            # Only add label once per run
            plot_label = label if segment_idx == 0 else None

            ax.plot(
                segment_positions[:, 0],
                segment_positions[:, 1],
                segment_positions[:, 2],
                color=color,
                alpha=0.3,
                linestyle="-",
                label=plot_label,
            )

            # Plot target location
            if hasattr(env_state_batch, "target_pos"):
                target_positions = np.asarray(env_state_batch.target_pos)
                target_pos = target_positions[start_idx, env_idx]
                target_key = tuple(np.round(target_pos, 3).tolist())
                if target_key not in plotted_targets:
                    ax.scatter(
                        target_pos[0],
                        target_pos[1],
                        target_pos[2],
                        color="orange",
                        marker="x",
                        s=100,
                    )
                    plotted_targets.add(target_key)
                    target_label_added = True

            # Plot obstacle if exists
            if hasattr(env_state_batch, "obstacle_pos"):
                obstacle_positions = np.asarray(env_state_batch.obstacle_pos)
                obstacle_pos = obstacle_positions[start_idx, env_idx]
                obstacle_key = tuple(np.round(obstacle_pos, 3).tolist())
                if obstacle_key not in plotted_obstacles:
                    obs_radius = 0.2
                    if env_params is not None and hasattr(env_params, "obstacle_radius"):
                        obs_radius = float(env_params.obstacle_radius)

                    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
                    x = obstacle_pos[0] + obs_radius * np.cos(u) * np.sin(v)
                    y = obstacle_pos[1] + obs_radius * np.sin(u) * np.sin(v)
                    z = obstacle_pos[2] + obs_radius * np.cos(v)

                    ax.plot_surface(
                        x, y, z,
                        color="yellow",
                        alpha=0.25,
                    )
                    plotted_obstacles.add(obstacle_key)

                    if not obstacle_label_added:
                        # Add proxy artist to legend for obstacle
                        ax.plot(
                            [], [],
                            color="yellow",
                            alpha=0.3,
                            linewidth=5,
                        )
                        obstacle_label_added = True

            sample_indices = list(range(0, segment_positions.shape[0], step_every))
            if sample_indices[-1] != segment_positions.shape[0] - 1:
                sample_indices.append(segment_positions.shape[0] - 1)

            for sample_idx in sample_indices:
                _draw_transparent_quadrotor(
                    ax,
                    segment_positions[sample_idx],
                    segment_attitudes[sample_idx],
                    alpha=0.16,
                    color=color,
                )



    ax.legend(loc="upper right", fontsize=34)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350)
    plt.close(fig)


def _get_render_frame_count(info_batch, max_frames):
    returned_episode = jax.device_get(info_batch["returned_episode"])
    returned_episode = jnp.asarray(returned_episode)

    if returned_episode.ndim == 0:
        return int(max_frames)

    first_env_returns = (
        returned_episode[:, 0] if returned_episode.ndim > 1 else returned_episode
    )
    done_steps = jnp.where(first_env_returns)[0]

    if done_steps.size == 0:
        return int(max_frames)

    return max(1, min(int(max_frames), int(done_steps[0])))


def run_eval_loop_python(
    config,
    eval_step_fn,
    eval_start_state,
    run_dir,
    eval_video_wrapper=None,
    obs_noise=0.0,
):
    eval_state = eval_start_state
    info_list = []
    env_state_list = []

    # We can JIT the noise addition to avoid performance drops
    @jax.jit
    def add_noise(state, noise_std):
        rng, noise_rng = jax.random.split(state.rng)
        
        def _add_noise(leaf, key):
            if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
                return leaf + jax.random.normal(key, leaf.shape) * noise_std
            return leaf
            
        leaves, treedef = jax.tree_util.tree_flatten(state.last_obs)
        rngs = jax.random.split(noise_rng, len(leaves))
        new_leaves = [_add_noise(l, r) for l, r in zip(leaves, rngs)]
        new_obs = jax.tree_util.tree_unflatten(treedef, new_leaves)
        
        return state._replace(last_obs=new_obs, rng=rng)

    print("Running evaluation step by step...")
    for i in range(config["NUM_STEPS"]):
        if obs_noise > 0.0:
            eval_state = add_noise(eval_state, obs_noise)

        eval_state, (info, env_state) = eval_step_fn(eval_state, None)
        info_list.append(info)
        env_state_list.append(env_state)

    env_state_list = env_state_list[:-1]
    info_list = info_list[:]
    info_batch = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *info_list)
    env_state_batch = jax.tree_util.tree_map(
        lambda *args: jnp.stack(args), *env_state_list
    )

    returned_mask = info_batch["returned_episode"].astype(jnp.float32)
    returned_returns = info_batch["returned_episode_returns"]
    num_returned = returned_mask.sum()
    episodic_return = float(
        jnp.where(
            num_returned > 0,
            (returned_returns * returned_mask).sum() / num_returned,
            jnp.nan,
        )
    )

    video_path = None
    if eval_video_wrapper is not None:
        video_dir = run_dir / "video_eval"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "eval_video.mp4"
        video_frames = _get_render_frame_count(info_batch, config["NUM_STEPS"])
        video_env_state_batch = jax.tree_util.tree_map(
            lambda x: x[:video_frames], env_state_batch
        )
        eval_video_wrapper.render_video(
            env_state_batch=video_env_state_batch,
            output_path=video_path,
            max_frames=video_frames,
            width=config.get("EVAL_RENDER_WIDTH", 640),
            height=config.get("EVAL_RENDER_HEIGHT", 480),
            fps=eval_video_wrapper.get_render_fps(),
        )

    metrics = {
        "episodic_return": episodic_return,
        "video_path": str(video_path) if video_path else None,
    }
    return metrics, env_state_batch, info_batch


def load_params_from_msgpack(train_state, msgpack_bytes, is_nnx=False):
    if hasattr(train_state.params, "to_pure_dict"):
        empty_pure_dict = train_state.params.to_pure_dict()
        loaded_pure_dict = serialization.from_bytes(empty_pure_dict, msgpack_bytes)
        return nnx.State(loaded_pure_dict)
    else:
        loaded_pure_dict = serialization.from_bytes(train_state.params, msgpack_bytes)
        if is_nnx:
            return nnx.State(loaded_pure_dict)
        return loaded_pure_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policies.")
    parser.add_argument(
        "run_dirs", nargs="+", type=str, help="Paths to the trained run directories"
    )
    parser.add_argument(
        "--episodes", type=int, default=4, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render video of the evaluation"
    )
    parser.add_argument(
        "--obs_noise",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added to observations",
    )
    parser.add_argument(
        "--state_noise",
        type=float,
        default=0.0,
        help="Bound for uniform noise added to the state/velocity",
    )
    parser.add_argument(
        "--same_start",
        action="store_true",
        help="If set, all parallel evaluation episodes will start from the exact same initial state",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="random",
        choices=["random", "cross", "corners"],
        help="Initial scenario for multi-agent environment ('cross' / 'corners' spawns 4 agents at 4 corners heading to opposite corners)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help="Custom labels for the runs in the legend (must match number of runs)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.run_dirs):
        print(
            f"Error: Number of --labels ({len(args.labels)}) must match number of run_dirs ({len(args.run_dirs)})."
        )
        sys.exit(1)

    env_name = None
    base_env = None
    env_params = None
    runs_data = []

    for i, run_dir_str in enumerate(args.run_dirs):
        run_dir = Path(run_dir_str)
        config_path = run_dir / "config.json"
        params_path = run_dir / "policy_params.msgpack"

        if not config_path.exists() or not params_path.exists():
            print(f"Could not find config.json or policy_params.msgpack in {run_dir}")
            sys.exit(1)

        with open(config_path, "r") as f:
            config = json.load(f)

        env_name = config["ENV_NAME"]
        base_env = ENV_REGISTRY[config["ENV_NAME"]](**config.get("ENV_KWARGS", {}))
        env_params = base_env.default_params
        
        replace_kwargs = {"is_testing": True}
        if hasattr(env_params, "state_noise_bound"):
            replace_kwargs["state_noise_bound"] = args.state_noise
        if hasattr(env_params, "scenario"):
            replace_kwargs["scenario"] = args.scenario
        
        env_params = env_params.replace(**replace_kwargs)

        # For evaluation we only need this many environments
        config["NUM_ENVS"] = args.episodes

        # We optionally enable rendering
        config["EVAL_ENABLED"] = True
        config["EVAL_RENDER_VIDEO"] = args.render
        config["EVAL_RENDER_MAX_FRAMES"] = config["NUM_STEPS"]

        algorithm = config.get("ALGORITHM", "ppo").lower()
        spec = ALGORITHM_REGISTRY[algorithm]

        env = spec.wrap_env(base_env, config)

        # Force a consistent seed for evaluation so runs are perfectly comparable
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)

        networks = spec.init_networks(config, env, env_params, init_rng)

        # Reload parameters into the train state
        msgpack_bytes = params_path.read_bytes()
        is_nnx = "graphdef" in networks
        networks["train_state"] = networks["train_state"].replace(
            params=load_params_from_msgpack(networks["train_state"], msgpack_bytes, is_nnx)
        )

        graphdef = networks.get("graphdef")
        rngs_state = networks.get("rngs_state", {})

        if hasattr(spec, "make_eval_step"):
            eval_step_fn = spec.make_eval_step(
                config, graphdef, env, env_params, rngs_state=rngs_state
            )
        else:
            eval_step_fn = _build_deterministic_eval_step(
                config, graphdef, rngs_state, env, env_params
            )
        
        if is_nnx:
            import inspect
            eval_step_fn_inner = nnx.jit(eval_step_fn)
            
            sig = inspect.signature(eval_step_fn)
            takes_model = len(sig.parameters) == 3

            if "network" in networks:
                model = networks["network"]
                # Update the model with loaded parameters
                params_to_load = networks["train_state"].params
                if "actor" in params_to_load:
                    params_to_load = params_to_load["actor"]
                nnx.update(model, params_to_load)
            elif takes_model:
                raise RuntimeError("eval_step_fn expects 'model' as first argument, but 'network' not found in networks")
            
            def eval_step_fn_jit(eval_state, _unused):
                if takes_model:
                    new_eval_state, (info, env_state) = eval_step_fn_inner(model, eval_state, _unused)
                else:
                    new_eval_state, (info, env_state) = eval_step_fn_inner(eval_state, _unused)
                return new_eval_state, (info, env_state)
        else:
            eval_step_fn_inner = jax.jit(eval_step_fn)
            def eval_step_fn_jit(eval_state, _unused):
                return eval_step_fn_inner(eval_state.train_state.apply_fn, eval_state, _unused)

        if args.same_start:
            class SameStartWrapper:
                def __init__(self, env):
                    self._env = env
                def __getattr__(self, name):
                    return getattr(self._env, name)
                def reset(self, rng, params):
                    first_rng = rng[0]
                    same_rngs = jnp.tile(first_rng, (rng.shape[0], 1))
                    return self._env.reset(same_rngs, params)
            
            env = SameStartWrapper(env)

        # Get runner state for evaluation.
        rng, reset_rng = jax.random.split(rng)
        eval_start_state = spec.init_runner_state(
            config, env, env_params, networks, reset_rng
        )

        eval_video_wrapper = EvalVideoWrapper(env) if args.render else None

        label = args.labels[i] if args.labels else run_dir.name
        print(
            f"Evaluating {algorithm.upper()} policy '{label}' from {run_dir.name} for {args.episodes} episodes in {config['ENV_NAME']} with obs_noise={args.obs_noise}..."
        )

        metrics, env_state_batch, info_batch = run_eval_loop_python(
            config=config,
            eval_step_fn=eval_step_fn_jit,
            eval_start_state=eval_start_state,
            run_dir=run_dir,
            eval_video_wrapper=eval_video_wrapper,
            obs_noise=args.obs_noise,
        )

        print(f"Evaluation Results for {label}:", metrics)
        runs_data.append((env_state_batch, info_batch, label))

        # Clear JAX caches to avoid metadata equality check crashes on EnvParams arrays
        jax.clear_caches()

    # After all runs are evaluated, plot combined
    if runs_data:
        first_run_dir = Path(args.run_dirs[0])
        plot_dir = first_run_dir.parent / "combined_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        error_plot_path = plot_dir / "combined_eval_error_trajectory.png"
        trajectory_plot_path = plot_dir / "combined_eval_trajectory_3d.png"

        print(f"Generating combined plots in {plot_dir}...")
        try:
            # Check if we are in Nav (2D) or Drone (3D)
            sample_state = runs_data[0][0]
            while hasattr(sample_state, "env_state"):
                sample_state = sample_state.env_state
            
            if hasattr(sample_state, "target"):
                trajectory_plot_path = plot_dir / "combined_eval_trajectory_2d.png"
                _save_combined_trajectory_2d_plot(runs_data, trajectory_plot_path, env_params=env_params)
                
                bar_plot_path = plot_dir / "combined_eval_collision_rate.png"
                _save_combined_collision_rate_bar_plot(runs_data, bar_plot_path)

                error_plot_path = plot_dir / "combined_eval_error_trajectory.png"
                _save_combined_multi_agent_error_trajectory_plot(runs_data, error_plot_path)
            else:
                _save_combined_error_trajectory_plot(
                    [(data[0], data[2]) for data in runs_data], error_plot_path
                )
                _save_combined_trajectory_3d_plot(runs_data, trajectory_plot_path, env_params=env_params)
        except Exception as e:
            print(f"Skipped plotting trajectories due to error: {e}")

        import numpy as np
        for env_state_batch, info_batch, label in runs_data:
            state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
            info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)
            while hasattr(state_batch, "env_state"):
                state_batch = state_batch.env_state
            
            model_state = getattr(state_batch, "state", state_batch)
            raw_positions = np.asarray(model_state.pos) if hasattr(model_state, "pos") else None
            raw_attitudes = np.asarray(model_state.attitude) if hasattr(model_state, "attitude") else None
            
            returned_episode = np.asarray(info_batch["returned_episode"])
            
            returns = np.where(returned_episode, info_batch["returned_episode_returns"], np.nan)
            valid_returns = returns[~np.isnan(returns)]
            mean_return = np.nanmean(valid_returns) if len(valid_returns) > 0 else 0.0
            std_return = np.nanstd(valid_returns) if len(valid_returns) > 0 else 0.0
            print(f"Stats for {label} - Mean Return: {mean_return:.2f} +- {std_return:.2f}")
            
            if "MPC_COSTS" in info_batch:
                costs = np.asarray(info_batch["MPC_COSTS"])
                print(f"Stats for {label} - Mean MPC Cost: {np.nanmean(costs):.4f}")
                
            if "collision" in info_batch:
                collisions = np.asarray(info_batch["collision"])
                num_done = np.sum(returned_episode)
                if num_done > 0:
                    collision_rate = np.sum(collisions) / num_done
                    print(f"Stats for {label} - Collision Rate: {collision_rate:.2%}")

            num_steps = raw_positions.shape[0] if raw_positions is not None else 0
            num_envs = raw_positions.shape[1] if (raw_positions is not None and raw_positions.ndim > 2) else 1

            if hasattr(state_batch, "target") and raw_positions is not None:
                # Report statistics for non-colliding target error trajectories
                clean_target_errors = []
                collisions_arr = np.asarray(info_batch["collision"]) if "collision" in info_batch else None
                term_positions = np.asarray(info_batch["terminal_pos"]) if "terminal_pos" in info_batch else raw_positions

                for env_idx in range(num_envs):
                    env_returns = (
                        returned_episode[:, env_idx]
                        if returned_episode.ndim > 1
                        else returned_episode
                    )
                    done_steps = np.where(env_returns)[0].tolist()
                    if len(done_steps) > 0:
                        first_done_idx = min(done_steps[0], len(raw_positions) - 1)
                        env_p = np.zeros((first_done_idx + 1, *raw_positions.shape[2:]), dtype=raw_positions.dtype)
                        env_p[:first_done_idx] = raw_positions[:first_done_idx, env_idx]
                        env_p[first_done_idx] = term_positions[first_done_idx, env_idx]
                        has_collided = (
                            bool(np.any(collisions_arr[: first_done_idx + 1, env_idx]))
                            if collisions_arr is not None
                            else False
                        )
                    else:
                        first_done_idx = len(raw_positions) - 1
                        env_p = raw_positions[:, env_idx]
                        has_collided = (
                            bool(np.any(collisions_arr[:, env_idx]))
                            if collisions_arr is not None
                            else False
                        )

                    if not has_collided:
                        tgt_arr = np.asarray(state_batch.target)
                        env_tgt = tgt_arr[0, env_idx] if tgt_arr.ndim == 4 else tgt_arr[env_idx]
                        agent_errs = np.linalg.norm(env_p - env_tgt[None, :, :], axis=-1)
                        clean_target_errors.append(agent_errs.mean(axis=-1))

                if len(clean_target_errors) > 0:
                    init_errs = [e[0] for e in clean_target_errors]
                    final_errs = [e[-1] for e in clean_target_errors]
                    mean_errs = [np.mean(e) for e in clean_target_errors]
                    print(
                        f"Stats for {label} - Non-colliding Episodes: "
                        f"{len(clean_target_errors)}/{num_envs} "
                        f"({len(clean_target_errors) / num_envs * 100:.1f}%)"
                    )
                    print(
                        f"Stats for {label} - Non-colliding Target Error [m]: "
                        f"Initial: {np.mean(init_errs):.3f} ± {np.std(init_errs):.3f} | "
                        f"Final: {np.mean(final_errs):.3f} ± {np.std(final_errs):.3f} | "
                        f"Trajectory Mean: {np.mean(mean_errs):.3f} ± {np.std(mean_errs):.3f}"
                    )
                else:
                    print(
                        f"Stats for {label} - Non-colliding Episodes: 0/{num_envs} (100.0% collision rate)"
                    )
            
            try:
                if "terminal_pos" in info_batch:
                    terminal_positions = np.asarray(info_batch["terminal_pos"])
                    terminal_attitudes = np.asarray(info_batch["terminal_attitude"]) if "terminal_attitude" in info_batch else None
                else:
                    terminal_positions = raw_positions
                    terminal_attitudes = raw_attitudes
                
                num_steps = raw_positions.shape[0]
                num_envs = raw_positions.shape[1] if raw_positions.ndim > 2 else 1
                
                positions_list = []
                attitudes_list = []
                
                for env_idx in range(num_envs):
                    env_returns = (
                        returned_episode[:, env_idx]
                        if returned_episode.ndim > 1
                        else returned_episode
                    )
                    done_steps = np.where(env_returns)[0].tolist()
                    
                    if len(done_steps) > 0:
                        first_done_idx = min(done_steps[0], len(raw_positions) - 1)
                        env_pos = np.zeros((first_done_idx + 1, *raw_positions.shape[2:]), dtype=raw_positions.dtype)
                        env_pos[:first_done_idx] = raw_positions[:first_done_idx, env_idx]
                        env_pos[first_done_idx] = terminal_positions[first_done_idx, env_idx]
                        
                        if raw_attitudes is not None:
                            env_att = np.zeros((first_done_idx + 1, *raw_attitudes.shape[2:]), dtype=raw_attitudes.dtype)
                            env_att[:first_done_idx] = raw_attitudes[:first_done_idx, env_idx]
                            env_att[first_done_idx] = terminal_attitudes[first_done_idx, env_idx]
                        else:
                            env_att = None
                    else:
                        env_pos = raw_positions[:, env_idx]
                        env_att = raw_attitudes[:, env_idx] if raw_attitudes is not None else None
                    
                    positions_list.append(env_pos)
                    if env_att is not None:
                        attitudes_list.append(env_att)
                
                trajectory_data = {
                    "position": positions_list,
                }
                if attitudes_list:
                    trajectory_data["attitude"] = attitudes_list
                    
                npy_path = plot_dir / f"{label}_trajectory.npy"
                np.save(npy_path, trajectory_data)
                print(f"Saved perturbed trajectory for '{label}' to {npy_path}")
            except Exception as e:
                print(f"Skipping saving .npy trajectory for '{label}' due to error: {e}")

        print("Done.")


if __name__ == "__main__":
    main()
