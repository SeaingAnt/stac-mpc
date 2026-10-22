"""Render publication-quality ghost and trail motion images for MultiAgentNav2DV2.

Usage Examples:
  # 1. Render from a trained checkpoint with 'cross' corner-swap scenario:
  pixi run python transformer_mpc/render_ghost_trail.py runs/multi_agent_nav_v2/MULTI_AGENT_DIFFMPC_TRANSFORMER_STAB_20261008-145256 --scenario cross --output ghost_cross.png

  # 2. Render from an existing saved trajectory .npy file:
  pixi run python transformer_mpc/render_ghost_trail.py --from_npy runs/multi_agent_nav_v2/combined_plots/MULTI_AGENT_DIFFMPC_TRANSFORMER_STAB_20261008-145256_trajectory.npy --output ghost_npy.png

  # 3. Render random scenario with 8 ghosts:
  pixi run python transformer_mpc/render_ghost_trail.py runs/multi_agent_nav_v2/MULTI_AGENT_DIFFMPC_TRANSFORMER_STAB_20261008-145256 --scenario random --num_ghosts 10 --output ghost_random.png
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Force CPU execution for evaluation to avoid GPU fragmentation issues
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Add project root and subdirectories to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "mpx"))
sys.path.append(os.path.join(root_dir, "transformer_mpc"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from env import ENV_REGISTRY
from algorithms import ALGORITHM_REGISTRY
from eval import load_params_from_msgpack, _build_deterministic_eval_step


def render_trajectory_ghost(
    env,
    states: Any,
    output_path: Optional[Path | str] = None,
    num_ghosts: int = 8,
    step_every: Optional[int] = None,
    show_lidar: bool = True,
    show_ghost_lidar: Optional[bool] = None,
    show_start: bool = True,
    show_agent_ids: bool = True,
    show_goal_radius: bool = True,
    show_axis: bool = False,
    show_arena_bounds: bool = True,
    ghost_alpha_min: float = 0.12,
    ghost_alpha_max: float = 0.55,
    final_alpha: float = 0.95,
    trail_linestyle: str = "-",
    trail_linewidth: float = 2.0,
    trail_alpha: float = 0.75,
    title: Optional[str] = "Multi-Agent Navigation",
    width: int = 800,
    height: int = 800,
    dpi: int = 200,
) -> np.ndarray:
    """Helper to render ghost and trail image directly using base environment."""
    base_env = env
    while hasattr(base_env, "_env"):
        base_env = base_env._env

    return base_env.render_ghost_and_trail(
        states=states,
        num_ghosts=num_ghosts,
        step_every=step_every,
        output_path=output_path,
        width=width,
        height=height,
        dpi=dpi,
        show_lidar=show_lidar,
        show_ghost_lidar=show_ghost_lidar,
        show_start=show_start,
        show_agent_ids=show_agent_ids,
        show_goal_radius=show_goal_radius,
        show_axis=show_axis,
        show_arena_bounds=show_arena_bounds,
        ghost_alpha_min=ghost_alpha_min,
        ghost_alpha_max=ghost_alpha_max,
        final_alpha=final_alpha,
        trail_linestyle=trail_linestyle,
        trail_linewidth=trail_linewidth,
        trail_alpha=trail_alpha,
        title=title,
    )


def rollout_policy(
    config: Dict[str, Any],
    run_dir: Path,
    scenario: str = "cross",
    num_agents: Optional[int] = None,
    seed: int = 42,
    max_steps: int = 100,
):
    """Run a single evaluation episode with the trained policy and return the trajectory."""
    env_name = config.get("ENV_NAME", "multi_agent_nav_v2")
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    if num_agents is not None:
        env_kwargs["num_agents"] = num_agents

    base_env = ENV_REGISTRY[env_name](**env_kwargs)
    env_params = base_env.default_params.replace(
        is_testing=True,
        scenario=scenario,
        max_steps_in_episode=max_steps,
    )

    config = dict(config)
    config["NUM_ENVS"] = 1
    config["EVAL_ENABLED"] = True

    algorithm = config.get("ALGORITHM", "ppo").lower()
    spec = ALGORITHM_REGISTRY[algorithm]

    env = spec.wrap_env(base_env, config)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    networks = spec.init_networks(config, env, env_params, init_rng)

    # Load trained weights
    params_path = run_dir / "policy_params.msgpack"
    if params_path.exists():
        msgpack_bytes = params_path.read_bytes()
        is_nnx = "graphdef" in networks
        networks["train_state"] = networks["train_state"].replace(
            params=load_params_from_msgpack(networks["train_state"], msgpack_bytes, is_nnx)
        )
    else:
        print(f"Warning: {params_path} not found. Running with initial/untrained parameters.")

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

    is_nnx = "graphdef" in networks
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
            raise RuntimeError("eval_step_fn expects model, but 'network' not found")

        def eval_step_fn_jit(eval_state, _unused):
            if takes_model:
                return eval_step_fn_inner(model, eval_state, _unused)
            return eval_step_fn_inner(eval_state, _unused)
    else:
        eval_step_fn_inner = jax.jit(eval_step_fn)

        def eval_step_fn_jit(eval_state, _unused):
            return eval_step_fn_inner(eval_state.train_state.apply_fn, eval_state, _unused)

    rng, reset_rng = jax.random.split(rng)
    eval_start_state = spec.init_runner_state(
        config, env, env_params, networks, reset_rng
    )

    current_eval_state = eval_start_state
    trajectory_states = []

    # Record initial state
    curr_s = current_eval_state.env_state
    trajectory_states.append(curr_s)

    for step_idx in range(max_steps):
        current_eval_state, (info, env_state) = eval_step_fn_jit(
            current_eval_state, None
        )

        # Check termination
        terminated = False
        if "returned_episode" in info:
            ret = np.asarray(jax.device_get(info["returned_episode"]))
            terminated = bool(np.any(ret))

        if terminated:
            # If auto-reset happened, grab true terminal_pos from info
            if "terminal_pos" in info:
                term_p = np.asarray(jax.device_get(info["terminal_pos"]))
                if term_p.ndim > 2:
                    term_p = term_p[0]
                # Replace pos in a copy or construct terminal NavState
                s_unwrapped = env_state
                while hasattr(s_unwrapped, "env_state"):
                    s_unwrapped = s_unwrapped.env_state
                last_target = getattr(s_unwrapped, "target", getattr(getattr(s_unwrapped, "state", s_unwrapped), "target", None))
                if last_target is not None:
                    last_target = np.asarray(jax.device_get(last_target))
                    if last_target.ndim > 2:
                        last_target = last_target[0]
                else:
                    last_target = np.full((base_env.num_agents, 2), 1.5)

                term_state = {
                    "pos": term_p,
                    "target": last_target,
                }
                trajectory_states.append(term_state)
            break
        else:
            trajectory_states.append(env_state)

    return base_env, trajectory_states


def main():
    parser = argparse.ArgumentParser(
        description="Render an image with ghost and trail motion for multi_agent_nav_v2."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default="runs/multi_agent_nav_v2/MULTI_AGENT_DIFFMPC_TRANSFORMER_STAB_20261008-145256",
        help="Path to trained run directory containing config.json and policy_params.msgpack",
    )
    parser.add_argument(
        "--from_npy",
        type=str,
        default=None,
        help="Optional path to a saved .npy trajectory file instead of running a policy",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="cross",
        choices=["cross", "corners", "random"],
        help="Initial layout scenario: 'cross' (corner swap), 'corners', or 'random'",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=4,
        help="Number of agents to render",
    )
    parser.add_argument(
        "--num_ghosts",
        type=int,
        default=8,
        help="Number of ghost silhouettes to draw along the trajectory",
    )
    parser.add_argument(
        "--step_every",
        type=int,
        default=None,
        help="Sample ghosts every N steps instead of uniform count",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (e.g. ghost_motion.png)",
    )
    parser.add_argument(
        "--no_lidar",
        action="store_true",
        help="Disable lidar rays (lidar is enabled by default to show local observations)",
    )
    parser.add_argument(
        "--hide_agent_ids",
        action="store_true",
        help="Hide agent ID numbers on robot bodies",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI resolution for saved figure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode initialization",
    )
    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="Show coordinate axes and grid (hidden by default for clean presentation)",
    )
    parser.add_argument(
        "--hide_arena_bounds",
        action="store_true",
        help="Hide dashed arena boundary rectangle when axis is removed",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    args = parser.parse_args()

    # Determine default output path
    if args.output is None:
        if args.from_npy:
            npy_p = Path(args.from_npy)
            out_path = npy_p.parent / f"{npy_p.stem}_ghost_lidar.png"
        elif args.run_dir:
            run_p = Path(args.run_dir)
            out_path = run_p / f"ghost_motion_lidar_{args.scenario}.png"
        else:
            out_path = Path("ghost_motion_lidar.png")
    else:
        out_path = Path(args.output)

    title = args.title
    if title is None:
        title = f"Multi-Agent Navigation"

    show_lidar = not args.no_lidar

    if args.from_npy:
        print(f"Loading trajectory from .npy file: {args.from_npy}...")
        raw_data = np.load(args.from_npy, allow_pickle=True)
        if isinstance(raw_data, np.ndarray) and raw_data.dtype == object and raw_data.ndim == 0:
            dict_data = raw_data.item()
            pos = dict_data.get("position", dict_data.get("pos"))
        else:
            pos = raw_data

        pos_arr = np.asarray(pos)
        if pos_arr.ndim == 4:
            pos_arr = pos_arr[0]  # (T, N, 2)
        elif pos_arr.ndim == 2:
            pos_arr = pos_arr[:, None, :]

        num_agents = pos_arr.shape[1]
        env = ENV_REGISTRY["multi_agent_nav_v2"](num_agents=num_agents)

        print(f"Rendering ghost motion for {num_agents} agents across {len(pos_arr)} steps...")
        render_trajectory_ghost(
            env=env,
            states=pos_arr,
            output_path=out_path,
            num_ghosts=args.num_ghosts,
            step_every=args.step_every,
            show_lidar=show_lidar,
            show_agent_ids=not args.hide_agent_ids,
            show_axis=args.show_axis,
            show_arena_bounds=not args.hide_arena_bounds,
            title=title,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
    else:
        run_path = Path(args.run_dir)
        config_path = run_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        print(f"Rolling out policy from {run_path.name} (scenario: {args.scenario})...")
        env, trajectory_states = rollout_policy(
            config=config,
            run_dir=run_path,
            scenario=args.scenario,
            num_agents=args.num_agents,
            seed=args.seed,
        )

        print(f"Rendered rollout with {len(trajectory_states)} states. Saving ghost & trail image to {out_path}...")
        render_trajectory_ghost(
            env=env,
            states=trajectory_states,
            output_path=out_path,
            num_ghosts=args.num_ghosts,
            step_every=args.step_every,
            show_lidar=show_lidar,
            show_agent_ids=not args.hide_agent_ids,
            show_axis=args.show_axis,
            show_arena_bounds=not args.hide_arena_bounds,
            title=title,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )

    print(f"Successfully generated image: {out_path}")


if __name__ == "__main__":
    main()
