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
from train import _build_deterministic_eval_step
from env.wrappers import EvalVideoWrapper

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


def _save_combined_error_trajectory_plot(runs_data, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, (ax_pos, ax_yaw) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for i, (env_state_batch, label) in enumerate(runs_data):
        env_state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)

        while hasattr(env_state_batch, "env_state"):
            env_state_batch = env_state_batch.env_state

        model_state = env_state_batch.state
        target_pos = np.asarray(env_state_batch.target_pos)
        target_yaw = np.asarray(env_state_batch.target_yaw)

        positions = np.asarray(model_state.pos)
        attitudes = np.asarray(model_state.attitude)

        pos_error = np.linalg.norm(positions - target_pos, axis=-1)

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

        time_axis = np.arange(pos_error.shape[0])
        pos_mean = pos_error.mean(axis=1)
        pos_std = pos_error.std(axis=1)
        yaw_mean = yaw_error.mean(axis=1)
        yaw_std = yaw_error.std(axis=1)

        color = STRONG_COLORS[i % len(STRONG_COLORS)]

        for episode_idx in range(pos_error.shape[1]):
            ax_pos.plot(
                time_axis,
                pos_error[:, episode_idx],
                color=color,
                alpha=0.1,
                linewidth=0.9,
            )
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
                alpha=0.5,
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
                        color="green",
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
                        alpha=0.15,
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
        noise = jax.random.normal(noise_rng, state.last_obs.shape) * noise_std
        return state._replace(last_obs=state.last_obs + noise, rng=rng)

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


def load_params_from_msgpack(train_state, msgpack_bytes):
    # Depending on the algorithm, the train_state params might just be the full state or a subset.
    # We will load the pure dict form from the msgpack file and match it to params.to_pure_dict()
    empty_pure_dict = train_state.params.to_pure_dict()
    loaded_pure_dict = serialization.from_bytes(empty_pure_dict, msgpack_bytes)
    return nnx.State(loaded_pure_dict)


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

        if env_name is None:
            env_name = config["ENV_NAME"]
            base_env = ENV_REGISTRY[config["ENV_NAME"]](**config.get("ENV_KWARGS", {}))
            env_params = base_env.default_params
            env_params = env_params.replace(is_testing=True)
        elif env_name != config["ENV_NAME"]:
            raise ValueError(
                f"Runs must come from the same environment. Found {env_name} and {config['ENV_NAME']}"
            )

        # For evaluation we only need this many environments
        config["NUM_ENVS"] = args.episodes

        # We optionally enable rendering
        config["EVAL_ENABLED"] = True
        config["EVAL_RENDER_VIDEO"] = args.render
        config["EVAL_RENDER_MAX_FRAMES"] = config["NUM_STEPS"]

        algorithm = config.get("ALGORITHM", "ppo").lower()
        spec = ALGORITHM_REGISTRY[algorithm]

        env = spec.wrap_env(base_env, config)

        rng = jax.random.PRNGKey(config.get("SEED", 30))
        rng, init_rng = jax.random.split(rng)

        networks = spec.init_networks(config, env, env_params, init_rng)

        # Reload parameters into the train state
        msgpack_bytes = params_path.read_bytes()
        networks["train_state"] = networks["train_state"].replace(
            params=load_params_from_msgpack(networks["train_state"], msgpack_bytes)
        )

        graphdef = networks["graphdef"]
        rngs_state = networks.get("rngs_state", {})

        eval_step_fn = _build_deterministic_eval_step(
            config, graphdef, rngs_state, env, env_params
        )
        eval_step_fn_jit = jax.jit(eval_step_fn)

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
        _save_combined_error_trajectory_plot(
            [(data[0], data[2]) for data in runs_data], error_plot_path
        )
        _save_combined_trajectory_3d_plot(runs_data, trajectory_plot_path, env_params=env_params)

        import numpy as np
        for env_state_batch, info_batch, label in runs_data:
            state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
            info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)
            while hasattr(state_batch, "env_state"):
                state_batch = state_batch.env_state
            
            model_state = state_batch.state
            raw_positions = np.asarray(model_state.pos)
            raw_attitudes = np.asarray(model_state.attitude)
            
            returned_episode = np.asarray(info_batch["returned_episode"])
            
            if "terminal_pos" in info_batch:
                terminal_positions = np.asarray(info_batch["terminal_pos"])
                terminal_attitudes = np.asarray(info_batch["terminal_attitude"])
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
                    first_done_idx = done_steps[0]
                    env_pos = np.zeros((first_done_idx + 1, 3), dtype=raw_positions.dtype)
                    env_pos[:first_done_idx] = raw_positions[:first_done_idx, env_idx]
                    env_pos[first_done_idx] = terminal_positions[first_done_idx, env_idx]
                    
                    env_att = np.zeros((first_done_idx + 1, 4), dtype=raw_attitudes.dtype)
                    env_att[:first_done_idx] = raw_attitudes[:first_done_idx, env_idx]
                    env_att[first_done_idx] = terminal_attitudes[first_done_idx, env_idx]
                else:
                    env_pos = raw_positions[:, env_idx]
                    env_att = raw_attitudes[:, env_idx]
                
                positions_list.append(env_pos)
                attitudes_list.append(env_att)
            
            trajectory_data = {
                "position": positions_list,
                "attitude": attitudes_list,
            }
            npy_path = plot_dir / f"{label}_trajectory.npy"
            np.save(npy_path, trajectory_data)
            print(f"Saved perturbed trajectory for '{label}' to {npy_path}")

        print("Done.")


if __name__ == "__main__":
    main()
