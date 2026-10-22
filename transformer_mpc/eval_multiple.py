import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Force CPU execution for evaluation as XLA GPU compilation of the
# unrolled MPC solver causes "Contracting dimension is too fragmented"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=3 --xla_gpu_force_compilation_parallelism=16"
)

# Add the project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "mpx"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from algorithms import ALGORITHM_REGISTRY
from env import ENV_REGISTRY
from env.wrappers import EvalVideoWrapper
from buffers import RunnerState

# Import plotting helpers from eval
from eval import (
    STRONG_COLORS,
    _build_deterministic_eval_step,
    _get_render_frame_count,
    run_eval_loop_python,
    _save_combined_multi_agent_error_trajectory_plot,
    _save_combined_collision_rate_bar_plot,
)


def parse_n_arguments(n_raw: List[Any]) -> List[int]:
    """Parse list of N arguments supporting both spaced (-N 2 4 6 8) and comma-separated (-N 2,4,6,8)."""
    parsed = []
    for item in n_raw:
        if isinstance(item, int):
            parsed.append(item)
        elif isinstance(item, str):
            for part in item.split(","):
                part = part.strip()
                if part:
                    try:
                        parsed.append(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid integer value for N: '{part}'")
    unique_n = sorted(list(set(parsed)))
    if not unique_n:
        raise ValueError("No valid values for N were provided.")
    return unique_n


def load_policy_params_for_n_agents(train_state, msgpack_bytes: bytes, num_agents: int, is_nnx: bool = False):
    """Safely load policy parameters into train_state when num_agents may differ from the trained model.
    
    In multi-agent CTDE:
    - The decentralized actor operates per-agent on local observations, with weights shared across agents.
    - All network weights (projections, transformer blocks, MLP layers) are invariant to N.
    - The critic (which depends on N * local_obs_dim + 1) is unused in evaluation.
    - actor_log_std (shape (N_train, action_dim)) is adapted to (num_agents, action_dim) if needed.
    """
    raw_state_dict = serialization.msgpack_restore(msgpack_bytes)

    if hasattr(train_state.params, "to_pure_dict"):
        template_dict = train_state.params.to_pure_dict()
    elif isinstance(train_state.params, dict):
        template_dict = train_state.params
    else:
        template_dict = train_state.params

    def _adapt_array(loaded_val, target_shape):
        arr = np.asarray(loaded_val)
        if arr.shape == target_shape:
            return arr
        if len(target_shape) == 2 and arr.ndim == 2 and arr.shape[1] == target_shape[1]:
            target_n = target_shape[0]
            curr_n = arr.shape[0]
            if target_n <= curr_n:
                return arr[:target_n]
            else:
                repeats = (target_n + curr_n - 1) // curr_n
                return np.tile(arr, (repeats, 1))[:target_n]
        return np.zeros(target_shape, dtype=arr.dtype)

    def _adapt_subdict(loaded_d, template_d):
        adapted = {}
        for k, v in template_d.items():
            if k in loaded_d:
                val = loaded_d[k]
                if isinstance(v, dict) and isinstance(val, dict):
                    adapted[k] = _adapt_subdict(val, v)
                elif hasattr(v, "shape"):
                    target_shape = v.shape
                    if hasattr(val, "shape") and val.shape != target_shape:
                        adapted[k] = _adapt_array(val, target_shape)
                    else:
                        adapted[k] = val
                else:
                    adapted[k] = val
            else:
                adapted[k] = v
        return adapted

    if "actor" in raw_state_dict and "actor" in template_dict:
        actor_state = raw_state_dict["actor"]
        template_actor = template_dict["actor"]

        # Adapt actor_log_std if present
        if "actor_log_std" in actor_state and "actor_log_std" in template_actor:
            loaded_log_std = actor_state["actor_log_std"]
            target_log_std = template_actor["actor_log_std"]

            if isinstance(loaded_log_std, dict) and "value" in loaded_log_std:
                target_shape = template_actor["actor_log_std"]["value"].shape
                actor_state["actor_log_std"]["value"] = _adapt_array(
                    loaded_log_std["value"], target_shape
                )
            elif hasattr(target_log_std, "shape"):
                actor_state["actor_log_std"] = _adapt_array(
                    loaded_log_std, target_log_std.shape
                )

        loaded_actor = serialization.from_state_dict(template_actor, actor_state)
        new_pure_dict = dict(template_dict)
        new_pure_dict["actor"] = loaded_actor
    else:
        adapted_dict = _adapt_subdict(raw_state_dict, template_dict)
        new_pure_dict = serialization.from_state_dict(template_dict, adapted_dict)

    if is_nnx:
        return nnx.State(new_pure_dict)
    return new_pure_dict


def save_trajectory_2d_plot_for_n(
    runs_data,
    output_path: Path,
    arena_size: float = 3.0,
    num_agents: Optional[int] = None,
):
    """Save 2D trajectories plot for multi-agent navigation with proper per-agent markers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([0, arena_size])
    ax.set_ylim([0, arena_size])
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)

    title_n = f" (N = {num_agents})" if num_agents is not None else ""
    ax.set_title(f"2D Trajectories across Evaluation Episodes{title_n}", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)

    linestyles = ['-', '--', '-.', ':']

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
        n_agents = positions.shape[2]
        style = linestyles[run_idx % len(linestyles)]
        num_envs = positions.shape[1]

        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, max(1, n_agents)))

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
                    (first_done_idx + 1, n_agents, 2), dtype=positions.dtype
                )
                env_pos[:first_done_idx] = positions[:first_done_idx, env_idx]
                env_pos[first_done_idx] = terminal_positions[first_done_idx, env_idx]
                env_target = target_pos[first_done_idx, env_idx]
            else:
                first_done_idx = len(positions) - 1
                env_pos = positions[:, env_idx]
                env_target = (
                    target_pos[-1, env_idx] if target_pos.ndim == 4 else target_pos[env_idx]
                )

            for agent_idx in range(n_agents):
                c = colors[agent_idx % len(colors)]
                agent_p = env_pos[:, agent_idx]
                ax.plot(
                    agent_p[:, 0],
                    agent_p[:, 1],
                    color=c,
                    alpha=0.6,
                    linewidth=1.8,
                    linestyle=style,
                    label=f"{label}" if (agent_idx == 0 and env_idx == 0) else "",
                )
                # Initial start
                ax.scatter(
                    agent_p[0, 0],
                    agent_p[0, 1],
                    color=c,
                    marker="o",
                    s=45,
                    edgecolors="black",
                    zorder=4,
                )
                # Final position
                ax.scatter(
                    agent_p[-1, 0],
                    agent_p[-1, 1],
                    color=c,
                    marker="x",
                    s=45,
                    zorder=4,
                )
                # Target position
                if env_target.ndim >= 2:
                    tgt_xy = env_target[agent_idx]
                    ax.scatter(
                        tgt_xy[0],
                        tgt_xy[1],
                        color=c,
                        marker="*",
                        s=80,
                        edgecolors="black",
                        zorder=5,
                    )

    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)
    print(f"Saved 2D trajectory plot to {output_path}")


def save_metrics_vs_n_plot(
    results_by_policy: Dict[str, Dict[int, Dict[str, Any]]],
    n_values: List[int],
    output_path: Path,
):
    """Generate 2x2 multi-panel figure comparing all policies across growing values of N."""
    import matplotlib.pyplot as plt

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_collision = axes[0, 0]
    ax_success = axes[0, 1]
    ax_error = axes[1, 0]
    ax_return = axes[1, 1]

    for p_idx, (label, n_dict) in enumerate(results_by_policy.items()):
        color = STRONG_COLORS[p_idx % len(STRONG_COLORS)]
        marker = markers[p_idx % len(markers)]

        xs = [n for n in n_values if n in n_dict]
        if not xs:
            continue

        collision_rates = [n_dict[n]["collision_rate"] for n in xs]
        success_rates = [n_dict[n]["success_rate"] for n in xs]
        final_errors = [n_dict[n]["final_target_error"] for n in xs]
        final_error_stds = [n_dict[n]["final_target_error_std"] for n in xs]
        returns = [n_dict[n]["mean_return"] for n in xs]
        return_stds = [n_dict[n]["std_return"] for n in xs]

        # 1. Collision Rate (%)
        ax_collision.plot(
            xs, collision_rates,
            color=color, marker=marker, linewidth=2.2, markersize=8, label=label
        )

        # 2. Success / Clean Episodes (%)
        ax_success.plot(
            xs, success_rates,
            color=color, marker=marker, linewidth=2.2, markersize=8, label=label
        )

        # 3. Final Target Error (m) with error bars
        ax_error.errorbar(
            xs, final_errors, yerr=final_error_stds,
            color=color, marker=marker, linewidth=2.2, markersize=8, capsize=4, label=label
        )

        # 4. Episodic Return with shaded std
        ax_return.plot(
            xs, returns,
            color=color, marker=marker, linewidth=2.2, markersize=8, label=label
        )
        returns_arr = np.array(returns)
        stds_arr = np.array(return_stds)
        ax_return.fill_between(
            xs, returns_arr - stds_arr, returns_arr + stds_arr,
            color=color, alpha=0.15
        )

    # Styling panel 1: Collision Rate
    ax_collision.set_title("Collision Rate vs. Number of Agents (N)", fontsize=13, fontweight="bold")
    ax_collision.set_xlabel("Number of Agents (N)", fontsize=12)
    ax_collision.set_ylabel("Collision Rate (%)", fontsize=12)
    ax_collision.set_xticks(n_values)
    ax_collision.set_ylim([-5, 105])
    ax_collision.grid(True, linestyle="--", alpha=0.4)
    ax_collision.legend(loc="best", fontsize=11)

    # Styling panel 2: Success Rate
    ax_success.set_title("Non-colliding Rate vs. Number of Agents (N)", fontsize=13, fontweight="bold")
    ax_success.set_xlabel("Number of Agents (N)", fontsize=12)
    ax_success.set_ylabel("Non-colliding Episodes (%)", fontsize=12)
    ax_success.set_xticks(n_values)
    ax_success.set_ylim([-5, 105])
    ax_success.grid(True, linestyle="--", alpha=0.4)
    ax_success.legend(loc="best", fontsize=11)

    # Styling panel 3: Target Error
    ax_error.set_title("Final Target Error vs. Number of Agents (N)", fontsize=13, fontweight="bold")
    ax_error.set_xlabel("Number of Agents (N)", fontsize=12)
    ax_error.set_ylabel("Final Target Error [m]", fontsize=12)
    ax_error.set_xticks(n_values)
    ax_error.set_ylim(bottom=0.0)
    ax_error.grid(True, linestyle="--", alpha=0.4)
    ax_error.legend(loc="best", fontsize=11)

    # Styling panel 4: Return
    ax_return.set_title("Episodic Return vs. Number of Agents (N)", fontsize=13, fontweight="bold")
    ax_return.set_xlabel("Number of Agents (N)", fontsize=12)
    ax_return.set_ylabel("Mean Return", fontsize=12)
    ax_return.set_xticks(n_values)
    ax_return.grid(True, linestyle="--", alpha=0.4)
    ax_return.legend(loc="best", fontsize=11)

    fig.suptitle("Zero-Shot Scaling Evaluation: Performance vs. Growing Swarm Size N", fontsize=15, y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined metrics vs. N plot to {output_path}")


def save_single_metric_vs_n_plot(
    results_by_policy: Dict[str, Dict[int, Dict[str, Any]]],
    n_values: List[int],
    metric_key: str,
    metric_name: str,
    ylabel: str,
    output_path: Path,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    """Generate standalone publication-quality curve for a single metric vs N."""
    import matplotlib.pyplot as plt

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    fig, ax = plt.subplots(figsize=(8, 6))

    for p_idx, (label, n_dict) in enumerate(results_by_policy.items()):
        color = STRONG_COLORS[p_idx % len(STRONG_COLORS)]
        marker = markers[p_idx % len(markers)]

        xs = [n for n in n_values if n in n_dict and metric_key in n_dict[n] and not np.isnan(n_dict[n][metric_key])]
        if not xs:
            continue

        ys = [n_dict[n][metric_key] for n in xs]
        ax.plot(
            xs, ys,
            color=color, marker=marker, linewidth=2.5, markersize=9, label=label
        )

    ax.set_title(f"{metric_name} vs. Number of Agents (N)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Agents (N)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(n_values)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {metric_name} plot to {output_path}")


def evaluate_policy_on_n(
    run_dir: Path,
    num_agents: int,
    args: argparse.Namespace,
    label: str,
    eval_video: bool = False,
):
    """Evaluate a single policy on multi-agent navigation with num_agents = N in fixed environment size."""
    config_path = run_dir / "config.json"
    params_path = run_dir / "policy_params.msgpack"

    with open(config_path, "r") as f:
        config = json.load(f)

    env_name = config.get("ENV_NAME", "multi_agent_nav_v2")
    if "multi_agent_nav" not in env_name:
        env_name = "multi_agent_nav_v2"

    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    env_kwargs["num_agents"] = num_agents

    base_env = ENV_REGISTRY[env_name](**env_kwargs)
    env_params = base_env.default_params

    replace_kwargs = {
        "is_testing": True,
        "num_agents": num_agents,
    }
    if hasattr(env_params, "state_noise_bound"):
        replace_kwargs["state_noise_bound"] = args.state_noise
    if hasattr(env_params, "scenario"):
        replace_kwargs["scenario"] = args.scenario

    env_params = env_params.replace(**replace_kwargs)

    config["NUM_ENVS"] = args.episodes
    config["EVAL_ENABLED"] = True
    config["EVAL_RENDER_VIDEO"] = eval_video
    config["EVAL_RENDER_MAX_FRAMES"] = config.get("NUM_STEPS", 100)

    algorithm = config.get("ALGORITHM", "diffmpc_transformer").lower()
    spec = ALGORITHM_REGISTRY[algorithm]

    env = spec.wrap_env(base_env, config)

    # Force consistent evaluation seed
    rng = jax.random.PRNGKey(args.seed + num_agents)
    rng, init_rng = jax.random.split(rng)

    networks = spec.init_networks(config, env, env_params, init_rng)

    # Reload parameters adapted for num_agents
    msgpack_bytes = params_path.read_bytes()
    is_nnx = "graphdef" in networks
    networks["train_state"] = networks["train_state"].replace(
        params=load_policy_params_for_n_agents(
            networks["train_state"], msgpack_bytes, num_agents=num_agents, is_nnx=is_nnx
        )
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
            params_to_load = networks["train_state"].params
            if "actor" in params_to_load:
                params_to_load = params_to_load["actor"]
            nnx.update(model, params_to_load)
        elif takes_model:
            raise RuntimeError("eval_step_fn expects 'model' as first argument, but 'network' not found")

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
            def __init__(self, wrapped_env):
                self._env = wrapped_env

            def __getattr__(self, name):
                return getattr(self._env, name)

            def reset(self, reset_rng, params):
                first_rng = reset_rng[0]
                same_rngs = jnp.tile(first_rng, (reset_rng.shape[0], 1))
                return self._env.reset(same_rngs, params)

        env = SameStartWrapper(env)

    rng, reset_rng = jax.random.split(rng)
    eval_start_state = spec.init_runner_state(
        config, env, env_params, networks, reset_rng
    )

    eval_video_wrapper = EvalVideoWrapper(env) if eval_video else None

    metrics, env_state_batch, info_batch = run_eval_loop_python(
        config=config,
        eval_step_fn=eval_step_fn_jit,
        eval_start_state=eval_start_state,
        run_dir=run_dir,
        eval_video_wrapper=eval_video_wrapper,
        obs_noise=args.obs_noise,
    )

    # Compute comprehensive metrics
    state_batch = jax.tree_util.tree_map(jax.device_get, env_state_batch)
    info_batch = jax.tree_util.tree_map(jax.device_get, info_batch)

    while hasattr(state_batch, "env_state"):
        state_batch = state_batch.env_state

    model_state = getattr(state_batch, "state", state_batch)
    raw_positions = np.asarray(model_state.pos) if hasattr(model_state, "pos") else None

    returned_episode = np.asarray(info_batch["returned_episode"])
    returns = np.where(returned_episode, info_batch["returned_episode_returns"], np.nan)
    valid_returns = returns[~np.isnan(returns)]
    mean_return = float(np.nanmean(valid_returns)) if len(valid_returns) > 0 else 0.0
    std_return = float(np.nanstd(valid_returns)) if len(valid_returns) > 0 else 0.0

    # Collision rate
    collision_rate = 0.0
    if "collision" in info_batch:
        collisions = np.asarray(info_batch["collision"])
        num_done = np.sum(returned_episode)
        if num_done > 0:
            collision_rate = float(np.sum(collisions) / num_done * 100.0)

    # Success rate
    success_rate = max(0.0, 100.0 - collision_rate)

    # MPC costs
    mean_mpc_cost = np.nan
    if "MPC_COSTS" in info_batch:
        costs = np.asarray(info_batch["MPC_COSTS"])
        mean_mpc_cost = float(np.nanmean(costs))

    # Clean non-colliding errors
    clean_target_errors = []
    num_envs = raw_positions.shape[1] if (raw_positions is not None and raw_positions.ndim > 2) else args.episodes
    collisions_arr = np.asarray(info_batch["collision"]) if "collision" in info_batch else None
    term_positions = (
        np.asarray(info_batch["terminal_pos"])
        if "terminal_pos" in info_batch
        else raw_positions
    )

    if hasattr(state_batch, "target") and raw_positions is not None:
        for env_idx in range(num_envs):
            env_returns = (
                returned_episode[:, env_idx]
                if returned_episode.ndim > 1
                else returned_episode
            )
            done_steps = np.where(env_returns)[0].tolist()
            if len(done_steps) > 0:
                first_done_idx = min(done_steps[0], len(raw_positions) - 1)
                env_p = np.zeros(
                    (first_done_idx + 1, *raw_positions.shape[2:]), dtype=raw_positions.dtype
                )
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
        init_err_mean = float(np.mean(init_errs))
        init_err_std = float(np.std(init_errs))
        final_err_mean = float(np.mean(final_errs))
        final_err_std = float(np.std(final_errs))
        traj_err_mean = float(np.mean(mean_errs))
        traj_err_std = float(np.std(mean_errs))
    else:
        init_err_mean = np.nan
        init_err_std = np.nan
        final_err_mean = np.nan
        final_err_std = np.nan
        traj_err_mean = np.nan
        traj_err_std = np.nan

    arena_size = float(getattr(env_params, "arena_size", 3.0))

    summary_stats = {
        "num_agents": num_agents,
        "policy": label,
        "arena_size": arena_size,
        "mean_return": mean_return,
        "std_return": std_return,
        "collision_rate": collision_rate,
        "success_rate": success_rate,
        "clean_episodes": len(clean_target_errors),
        "total_episodes": num_envs,
        "init_target_error": init_err_mean,
        "init_target_error_std": init_err_std,
        "final_target_error": final_err_mean,
        "final_target_error_std": final_err_std,
        "traj_target_error": traj_err_mean,
        "traj_target_error_std": traj_err_std,
        "mean_mpc_cost": mean_mpc_cost,
    }

    # Clear JAX caches
    jax.clear_caches()

    return summary_stats, env_state_batch, info_batch


def print_ascii_summary_table(results_by_policy: Dict[str, Dict[int, Dict[str, Any]]], n_values: List[int]):
    """Print an attractive summary table to the console."""
    print("\n" + "=" * 90)
    print(f"{'EVALUATION SUMMARY ACROSS GROWING NUMBER OF AGENTS (N)':^90}")
    print("=" * 90)

    header = f"{'Policy':<28} | {'N':>3} | {'Coll Rate (%)':>13} | {'Success (%)':>11} | {'Final Err (m)':>15} | {'Mean Return':>13}"
    print(header)
    print("-" * 90)

    for label, n_dict in results_by_policy.items():
        for n in n_values:
            if n in n_dict:
                stats = n_dict[n]
                c_rate = f"{stats['collision_rate']:.1f}%"
                s_rate = f"{stats['success_rate']:.1f}%"
                if not np.isnan(stats['final_target_error']):
                    err_str = f"{stats['final_target_error']:.3f} ± {stats['final_target_error_std']:.2f}"
                else:
                    err_str = "N/A (100% coll)"
                ret_str = f"{stats['mean_return']:.1f} ± {stats['std_return']:.1f}"
                print(f"{label[:28]:<28} | {n:>3} | {c_rate:>13} | {s_rate:>11} | {err_str:>15} | {ret_str:>13}")
        print("-" * 90)
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent V2 policies across a growing number of agents N in fixed environment size."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Paths to the trained run directories to evaluate",
    )
    parser.add_argument(
        "-N", "--num_agents", "--N", "--agents",
        nargs="+",
        default=[2, 4, 6, 8],
        help="List of agent counts to evaluate (e.g. -N 2 4 6 8 or -N 2,4,6,8). Default: [2, 4, 6, 8]",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate for each N (default: 10)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="random",
        choices=["random", "cross", "corners"],
        help="Initial spawn scenario ('random', 'cross', 'corners'). Default: random",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render video of the evaluation for each N and policy",
    )
    parser.add_argument(
        "--obs_noise",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise added to observations (default: 0.0)",
    )
    parser.add_argument(
        "--state_noise",
        type=float,
        default=0.0,
        help="Bound for uniform noise added to the state/velocity (default: 0.0)",
    )
    parser.add_argument(
        "--same_start",
        action="store_true",
        help="If set, all parallel evaluation episodes start from the exact same initial state",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        help="Custom labels for the runs in the legend and tables (must match number of run_dirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory for saving plots and results (default: <first_run_dir>/eval_multiple_N)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation (default: 42)",
    )

    args = parser.parse_args()

    # Parse N arguments
    n_values = parse_n_arguments(args.num_agents)
    print(f"Evaluating across N = {n_values} (fixed environment size)")

    if args.labels and len(args.labels) != len(args.run_dirs):
        print(
            f"Error: Number of --labels ({len(args.labels)}) must match number of run_dirs ({len(args.run_dirs)})."
        )
        sys.exit(1)

    labels = args.labels if args.labels else [Path(rd).name for rd in args.run_dirs]

    first_run_dir = Path(args.run_dirs[0])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = first_run_dir.parent / "eval_multiple_N"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results and plots will be saved to: {output_dir}")

    # Check existence of run files
    valid_runs = []
    for run_dir_str, label in zip(args.run_dirs, labels):
        run_dir = Path(run_dir_str)
        config_path = run_dir / "config.json"
        params_path = run_dir / "policy_params.msgpack"
        if not config_path.exists() or not params_path.exists():
            print(f"Error: Could not find config.json or policy_params.msgpack in {run_dir}")
            sys.exit(1)
        valid_runs.append((run_dir, label))

    results_by_policy: Dict[str, Dict[int, Dict[str, Any]]] = {label: {} for _, label in valid_runs}
    runs_data_by_n: Dict[int, List[Any]] = {n: [] for n in n_values}

    # Loop over each N in growing order
    for n in n_values:
        print(f"\n==================== EVALUATING SWARM SIZE N = {n} ====================")
        n_plot_dir = output_dir / f"N_{n}"
        n_plot_dir.mkdir(parents=True, exist_ok=True)

        for run_dir, label in valid_runs:
            print(f"\n---> Evaluating Policy '{label}' on N = {n} ({args.episodes} episodes)...")
            try:
                stats, env_state_batch, info_batch = evaluate_policy_on_n(
                    run_dir=run_dir,
                    num_agents=n,
                    args=args,
                    label=label,
                    eval_video=args.render,
                )
                results_by_policy[label][n] = stats
                runs_data_by_n[n].append((env_state_batch, info_batch, label))

                print(
                    f"Result [{label} | N={n}]: "
                    f"Collision Rate: {stats['collision_rate']:.1f}% | "
                    f"Success Rate: {stats['success_rate']:.1f}% | "
                    f"Mean Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f} | "
                    f"Final Target Error: {stats['final_target_error']:.3f} m"
                )
            except Exception as e:
                print(f"Error evaluating policy '{label}' on N={n}: {e}")
                import traceback
                traceback.print_exc()

        # Save individual per-N trajectory & error plots in fixed arena
        if runs_data_by_n[n]:
            try:
                traj_2d_path = n_plot_dir / "combined_eval_trajectory_2d.png"
                save_trajectory_2d_plot_for_n(
                    runs_data_by_n[n],
                    traj_2d_path,
                    arena_size=3.0,
                    num_agents=n,
                )

                bar_path = n_plot_dir / "combined_eval_collision_rate.png"
                _save_combined_collision_rate_bar_plot(runs_data_by_n[n], bar_path)

                err_path = n_plot_dir / "combined_eval_error_trajectory.png"
                _save_combined_multi_agent_error_trajectory_plot(runs_data_by_n[n], err_path)
            except Exception as e:
                print(f"Warning: Failed to generate per-N plots for N={n}: {e}")

    # Generate overarching comparison plots across all N values
    print("\nGenerating comparative plots across growing N...")
    combined_plot_path = output_dir / "metrics_vs_N.png"
    save_metrics_vs_n_plot(results_by_policy, n_values, combined_plot_path)

    save_single_metric_vs_n_plot(
        results_by_policy, n_values, "collision_rate", "Collision Rate", "Collision Rate (%)",
        output_dir / "collision_rate_vs_N.png", y_min=-5, y_max=105
    )
    save_single_metric_vs_n_plot(
        results_by_policy, n_values, "success_rate", "Non-Colliding Success Rate", "Success Rate (%)",
        output_dir / "success_rate_vs_N.png", y_min=-5, y_max=105
    )
    save_single_metric_vs_n_plot(
        results_by_policy, n_values, "final_target_error", "Final Target Error", "Target Error [m]",
        output_dir / "target_error_vs_N.png", y_min=0.0
    )
    save_single_metric_vs_n_plot(
        results_by_policy, n_values, "mean_return", "Episodic Return", "Mean Return",
        output_dir / "return_vs_N.png"
    )

    # Check if MPC cost was logged
    has_mpc_cost = any(
        not np.isnan(stats.get("mean_mpc_cost", np.nan))
        for p_dict in results_by_policy.values()
        for stats in p_dict.values()
    )
    if has_mpc_cost:
        save_single_metric_vs_n_plot(
            results_by_policy, n_values, "mean_mpc_cost", "Mean MPC Cost", "MPC Cost",
            output_dir / "mpc_cost_vs_N.png"
        )

    # Save summary JSON
    json_path = output_dir / "summary_metrics_vs_N.json"
    with open(json_path, "w") as f:
        json.dump(results_by_policy, f, indent=2)
    print(f"Saved numerical metrics JSON to {json_path}")

    # Save summary CSV
    csv_path = output_dir / "summary_metrics_vs_N.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Policy",
            "N",
            "Collision_Rate_Pct",
            "Success_Rate_Pct",
            "Clean_Episodes",
            "Total_Episodes",
            "Mean_Return",
            "Std_Return",
            "Final_Target_Error_m",
            "Std_Final_Target_Error_m",
            "Trajectory_Mean_Error_m",
            "Mean_MPC_Cost",
        ])
        for label, n_dict in results_by_policy.items():
            for n in n_values:
                if n in n_dict:
                    s = n_dict[n]
                    writer.writerow([
                        label,
                        n,
                        f"{s['collision_rate']:.2f}",
                        f"{s['success_rate']:.2f}",
                        s["clean_episodes"],
                        s["total_episodes"],
                        f"{s['mean_return']:.2f}",
                        f"{s['std_return']:.2f}",
                        f"{s['final_target_error']:.4f}" if not np.isnan(s['final_target_error']) else "N/A",
                        f"{s['final_target_error_std']:.4f}" if not np.isnan(s['final_target_error_std']) else "N/A",
                        f"{s['traj_target_error']:.4f}" if not np.isnan(s['traj_target_error']) else "N/A",
                        f"{s['mean_mpc_cost']:.4f}" if not np.isnan(s['mean_mpc_cost']) else "N/A",
                    ])
    print(f"Saved numerical metrics CSV to {csv_path}")

    # Print summary ASCII table
    print_ascii_summary_table(results_by_policy, n_values)
    print(f"All evaluations complete! Visualizations and tables are ready in:\n{output_dir}")


if __name__ == "__main__":
    main()
