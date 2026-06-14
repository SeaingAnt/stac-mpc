#!/usr/bin/env python3
"""Unified training entry point for PPO-family algorithms.

Usage:
    # Default config (ppo)
    python train.py

    # Select algorithm via Hydra override
    python train.py ++ALGORITHM=sppo_aug

    # Override environment and parameters
    python train.py ++ALGORITHM=ppo env=drone_v0 SEED=42
    python train.py ++ALGORITHM=sppo_aug env=drone_v1 LR=0.0005
"""

import os

import json
import importlib
import re

import time
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings(
    "ignore",
    message="overflow encountered in cast",
    module=r"jax\._src\.abstract_arrays",
    category=RuntimeWarning,
)

# Environment setup before JAX import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=16 --xla_cpu_enable_fast_math=true --xla_gpu_enable_fast_min_max=true "
)
# Note that setting xla_gpu_autotune_level=1 greatly improves performance, but causes non-determinism even when seeding the rng.
# For reproducibility in experiments, we set it to 0 (no autotuning).
# For final training runs where speed is more important than exact reproducibility, setting it to 1 is recommended.
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_autotune_level=1 --xla_gpu_force_compilation_parallelism=8 --xla_cpu_enable_fast_math=true --xla_gpu_enable_fast_min_max=true "
# )
# os.environ["JAX_PLATFORMS"] = "cuda,cpu"
import jax
import jax.numpy as jnp

import hydra
from flax import nnx, serialization
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None

from algorithms import ALGORITHM_REGISTRY, AlgorithmSpec
from buffers import UpdateState, RunnerState
from env import ENV_REGISTRY
from env.wrappers import EvalVideoWrapper
from utils import (
    update_latest_symlink,
    shuffle_batch,
)

_timing_state: dict = {"last_time": None}


def log_metrics(
    writer,
    losses,
    metric,
    targets,
    traj_batch_value,
    timestep,
    num_steps,
    num_envs,
    extra_losses=None,
    wandb_run=None,
):
    """Create logging callback for training metrics."""
    targets_var = jnp.var(targets)
    explained_var = 1 - jnp.var(targets - traj_batch_value) / (targets_var + 1e-8)

    def log_callback(losses, metric, explained_var, timestep, extra_losses):
        returned_mask = metric["returned_episode"]

        def _safe_mean(values):
            return jnp.where(values.size > 0, values.mean(), jnp.nan)

        episodic_return = _safe_mean(metric["returned_episode_returns"][returned_mask])
        episodic_return_true = _safe_mean(
            metric["returned_episode_returns_true"][returned_mask]
        )
        episodic_length = _safe_mean(metric["returned_episode_lengths"][returned_mask])

        writer.add_scalar("train/total_loss", float(losses["total"]), int(timestep))
        writer.add_scalar("train/value_loss", float(losses["value"]), int(timestep))
        writer.add_scalar("train/policy_loss", float(losses["actor"]), int(timestep))
        writer.add_scalar("train/entropy", float(losses["entropy"]), int(timestep))
        writer.add_scalar(
            "returns/episodic_return", float(episodic_return), int(timestep)
        )
        writer.add_scalar(
            "returns/episodic_return_true", float(episodic_return_true), int(timestep)
        )
        writer.add_scalar(
            "returns/episodic_length", float(episodic_length), int(timestep)
        )

        writer.add_scalar(
            "train/old_approx_kl", float(losses["old_approx_kl"]), int(timestep)
        )
        writer.add_scalar("train/approx_kl", float(losses["approx_kl"]), int(timestep))
        writer.add_scalar("train/clipfrac", float(losses["clipfrac"]), int(timestep))
        writer.add_scalar(
            "train/explained_variance", float(explained_var), int(timestep)
        )

        log_payload = {
            "train/total_loss": float(losses["total"]),
            "train/value_loss": float(losses["value"]),
            "train/policy_loss": float(losses["actor"]),
            "train/entropy": float(losses["entropy"]),
            "returns/episodic_return": float(episodic_return),
            "returns/episodic_return_true": float(episodic_return_true),
            "returns/episodic_length": float(episodic_length),
            "train/old_approx_kl": float(losses["old_approx_kl"]),
            "train/approx_kl": float(losses["approx_kl"]),
            "train/clipfrac": float(losses["clipfrac"]),
            "train/explained_variance": float(explained_var),
            "timestep": int(timestep),
        }

        # Wall-clock timing via Python-side mutable state
        now = time.time()
        if _timing_state["last_time"] is not None:
            elapsed = now - _timing_state["last_time"]
            steps_per_second = (num_steps * num_envs) / elapsed
            print(
                f"step={int(timestep)}  {elapsed:.2f}s  ({steps_per_second:.0f} steps/s)"
            )
            writer.add_scalar("perf/steps_per_second", steps_per_second, int(timestep))
            log_payload["perf/steps_per_second"] = float(steps_per_second)
        else:
            print(f"step={int(timestep)}  (first update, timing starts now)")
        _timing_state["last_time"] = now

        if extra_losses:
            for name, val in extra_losses.items():
                writer.add_scalar(f"train/{name}", float(val), int(timestep))
                log_payload[f"train/{name}"] = float(val)

        if wandb_run is not None:
            wandb_run.log(log_payload, step=int(timestep))

    jax.debug.callback(
        log_callback, losses, metric, explained_var, timestep, extra_losses
    )


def debug_callback(config, metric):
    """Debug callback for printing training progress."""

    def callback(info):
        returned_mask = info["returned_episode"]
        return_values = info["returned_episode_returns"][returned_mask]
        timesteps = info["timestep"][returned_mask] * config["NUM_ENVS"]

        if return_values.size == 0:
            print("global step=unknown, no completed episodes yet")
            return

        print(
            f"global step={timesteps[-1]}, mean episodic return={return_values.mean()}"
        )

    jax.debug.callback(callback, metric)


def make_train(
    config,
    spec: AlgorithmSpec,
    writer: SummaryWriter,
    init_rng,
    run_dir,
    wandb_run=None,
):
    """Create unified training function for PPO-family algorithms.

    Args:
        config: Training configuration dict
        spec: Algorithm specification
        writer: TensorBoard writer
        init_rng: PRNG key used for network initialisation (concrete, not traced)

    Returns:
        JIT-compilable train function
    """

    # Setup derived config values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = ENV_REGISTRY[config["ENV_NAME"]](**config.get("ENV_KWARGS", {}))
    env_params = env.default_params
    env = spec.wrap_env(env, config)

    # INIT NETWORKS outside JIT (NNX needs concrete rngs)
    networks = spec.init_networks(
        config,
        env,
        env_params,
        init_rng,
    )
    graphdef = networks["graphdef"]
    rngs_state = networks.get("rngs_state", {})

    # Create algorithm-specific functions (closures over graphdef)
    state_template = networks.get("state_template")
    loss_fn = spec.make_loss_fn(config, graphdef, rngs_state, state_template, env=env, env_params=env_params, networks=networks)
    collect_fn = spec.make_collect_fn(config, env, env_params, networks)
    update_fn = spec.make_update_fn(config, loss_fn, shuffle_batch, networks)
    eval_step_fn = _build_deterministic_eval_step(
        config, graphdef, rngs_state, env, env_params
    )
    eval_video_wrapper = EvalVideoWrapper(env)

    total_steps = config["TOTAL_TIMESTEPS"]

    eval_during_training_enabled = bool(
        config.get("EVAL_DURING_TRAINING_ENABLED", False)
    )
    eval_every_updates = int(config.get("EVAL_EVERY_UPDATES", 0))

    def train(rng):
        # INIT RUNNER STATE (algorithm handles state structure and RNG splitting)
        runner_state = spec.init_runner_state(config, env, env_params, networks, rng)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):

            # COLLECT TRAJECTORIES
            collect_state, traj_batch = jax.lax.scan(
                collect_fn, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE GAE
            gae_results, collect_state = spec.calculate_gae(
                config,
                traj_batch,
                networks,
                collect_state,
            )

            # UPDATE NETWORK
            gae_extras = {
                k: v
                for k, v in gae_results.items()
                if k not in ("advantages", "targets")
            }
            update_state = UpdateState(
                train_state=collect_state.train_state,
                traj_batch=traj_batch,
                advantages=gae_results["advantages"],
                targets=gae_results["targets"],
                rng=collect_state.rng,
                extras={**collect_state.extras, **gae_extras},
            )
            update_state, loss_info = jax.lax.scan(
                update_fn, update_state, None, config["UPDATE_EPOCHS"]
            )

            # Extract losses
            losses, extra_losses = spec.extract_losses(loss_info)

            metric = traj_batch.info
            if config.get("DEBUG", False):
                debug_callback(config, metric)

            # Logging (timing tracked via Python-side _timing_state)
            current_timestep = metric["timestep"].mean() * config["NUM_ENVS"]
            log_metrics(
                writer,
                losses,
                metric,
                gae_results["targets"],
                traj_batch.value,
                current_timestep,
                num_steps=config["NUM_STEPS"],
                num_envs=config["NUM_ENVS"],
                extra_losses=extra_losses,
                wandb_run=wandb_run,
            )

            # Optional periodic evaluation loop from inside training scan.
            if eval_during_training_enabled and eval_every_updates > 0:
                should_eval = ((update_idx + 1) % eval_every_updates) == 0

                def _periodic_eval_callback(
                    update_idx,
                    timestep,
                    should_eval,
                    eval_start_state,
                ):
                    if not bool(should_eval):
                        return
                    update_num = int(update_idx) + 1
                    timestep_int = int(timestep)

                    # debug.callback materializes values on host; move eval state back to
                    # a single device before calling JAX scans in evaluation.
                    try:
                        target_device = jax.devices()[0]

                        def _to_device(x):
                            if isinstance(x, jax.Array):
                                return jax.device_put(x, target_device)
                            if hasattr(x, "dtype"):
                                return jax.device_put(jnp.asarray(x), target_device)
                            return x

                        eval_start_state_dev = jax.tree.map(
                            _to_device, eval_start_state
                        )

                        run_evaluation_loop(
                            config,
                            eval_step_fn,
                            eval_start_state_dev,
                            eval_video_wrapper,
                            run_dir,
                            writer=writer,
                            wandb_run=wandb_run,
                            step=timestep_int,
                            trigger=f"during_training_update_{update_num}",
                        )
                    except Exception as exc:
                        # Keep training alive if periodic eval fails.
                        print(f"Skipping periodic eval at update {update_num}: {exc}")

                jax.debug.callback(
                    _periodic_eval_callback,
                    update_idx,
                    current_timestep,
                    should_eval,
                    collect_state,
                )

            # Rebuild runner state with updated train_state(s)
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=collect_state.env_state,
                last_obs=collect_state.last_obs,
                rng=update_state.rng,
                extras={**collect_state.extras, **update_state.extras},
            )
            return runner_state, None

        # Run training loop
        update_indices = jnp.arange(config["NUM_UPDATES"])
        runner_state, _ = jax.lax.scan(_update_step, runner_state, update_indices)
        return runner_state

    return train


def _policy_dist_from_model(model, obs, rng, **kwargs):
    """Return policy distribution while supporting model signatures with/without rng."""
    try:
        model_out = model(obs, rng_key=rng, **kwargs)
    except TypeError:
        try:
            model_out = model(obs, rng=rng, **kwargs)
        except TypeError:
            try:
                model_out = model(obs, rng, **kwargs)
            except TypeError:
                model_out = model(obs, **kwargs)
    return model_out[0]


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

    def _eval_step(eval_state: RunnerState, _unused):
        rng, model_rng = jax.random.split(eval_state.rng)
        model = nnx.merge(graphdef, eval_state.train_state.params, rngs_state)

        kwargs = {}
        if "DIFFMPC" in config.get("ALGORITHM", "").upper():
            kwargs["physical_state"] = get_physical_state(eval_state.env_state)

        pi = _policy_dist_from_model(model, eval_state.last_obs, model_rng, **kwargs)
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


def _run_deterministic_eval_rollout(config, eval_start_state, eval_step_fn):
    """Run deterministic rollout for NUM_STEPS and return episodic return mean."""
    _final_eval_state, (info_batch, env_state_batch) = jax.lax.scan(
        eval_step_fn,
        eval_start_state,
        None,
        config["NUM_STEPS"],
    )

    returned_mask = info_batch["returned_episode"].astype(jnp.float32)
    returned_returns = info_batch["returned_episode_returns"]
    num_returned = returned_mask.sum()
    episodic_return = jnp.where(
        num_returned > 0,
        (returned_returns * returned_mask).sum() / num_returned,
        jnp.nan,
    )
    return episodic_return, env_state_batch


def _render_eval_video(
    config, eval_video_wrapper, env_state_batch, run_dir, trigger, step, wandb_run
):
    """Render evaluation trajectory via wrapper/environment and optionally log to W&B."""
    if not config.get("EVAL_RENDER_VIDEO", False):
        return None

    if trigger.startswith("during_training") and not config.get(
        "EVAL_RENDER_DURING_TRAINING", False
    ):
        return None

    if eval_video_wrapper is None:
        return None

    render_height = int(config.get("EVAL_RENDER_HEIGHT", 480))
    render_width = int(config.get("EVAL_RENDER_WIDTH", 640))
    render_fps = int(eval_video_wrapper.get_render_fps(default_fps=30))
    max_frames = int(config.get("EVAL_RENDER_MAX_FRAMES", config["NUM_STEPS"]))
    max_frames = max(1, min(max_frames, int(config["NUM_STEPS"])))

    safe_trigger = re.sub(r"[^A-Za-z0-9_.-]+", "_", trigger)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"eval_{safe_trigger}_step_{int(step)}.mp4"

    rendered_path = eval_video_wrapper.render_video(
        env_state_batch=env_state_batch,
        output_path=video_path,
        max_frames=max_frames,
        fps=render_fps,
        width=render_width,
        height=render_height,
    )
    return rendered_path


def run_evaluation_loop(
    config,
    eval_step_fn,
    eval_start_state,
    eval_video_wrapper,
    run_dir,
    writer=None,
    wandb_run=None,
    step=None,
    trigger="post_training",
):
    """Run deterministic in-process evaluation loop and log episodic return."""
    if not config.get("EVAL_ENABLED", True):
        print("Evaluation disabled (EVAL_ENABLED=false).")
        return []

    eval_records = []
    episodic_return, env_state_batch = _run_deterministic_eval_rollout(
        config,
        eval_start_state,
        eval_step_fn,
    )
    episodic_return_f = float(episodic_return)

    record = {
        "trigger": trigger,
        "episodic_return": episodic_return_f,
    }
    eval_records.append(record)

    log_step = int(step) if step is not None else 0
    if writer is not None:
        writer.add_scalar("eval/episodic_return", episodic_return_f, log_step)

    video_path = _render_eval_video(
        config,
        eval_video_wrapper,
        env_state_batch,
        run_dir,
        trigger,
        log_step,
        wandb_run,
    )

    if wandb_run is not None:
        log_payload = {
            "eval/episodic_return": episodic_return_f,
            "eval/trigger": trigger,
            "timestep": log_step,
        }
        if video_path is not None:
            render_fps = int(eval_video_wrapper.get_render_fps(default_fps=30))
            print
            log_payload["eval/video"] = wandb.Video(
                str(video_path), fps=render_fps, format="mp4"
            )
            record["video_path"] = str(video_path)

        wandb_run.log(log_payload, step=log_step)

    eval_summary_path = run_dir / "evaluation_summary.json"
    all_records = []
    if eval_summary_path.exists():
        try:
            previous_records = json.loads(eval_summary_path.read_text())
            if isinstance(previous_records, list):
                all_records.extend(previous_records)
        except json.JSONDecodeError:
            pass
    all_records.extend(eval_records)

    eval_summary_path.write_text(json.dumps(all_records, indent=2))
    print(f"Saved evaluation summary to: {eval_summary_path}")
    return eval_records


def init_wandb(config, run_dir, algorithm):
    """Initialize Weights & Biases run if enabled in config."""
    if not config.get("WANDB_ENABLED", False):
        return None

    if wandb is None:
        print(
            "WANDB_ENABLED=true but wandb is not installed. Continuing without W&B logging."
        )
        return None

    return wandb.init(
        project=config.get("WANDB_PROJECT", "nphm-jax"),
        entity=config.get("WANDB_ENTITY") or None,
        name=config.get("WANDB_RUN_NAME") or run_dir.name,
        tags=config.get("WANDB_TAGS")
        or [algorithm, config.get("ENV_NAME", "unknown-env")],
        dir=str(run_dir),
        config=config,
        sync_tensorboard=bool(config.get("WANDB_SYNC_TENSORBOARD", True)),
        save_code=bool(config.get("WANDB_SAVE_CODE", False)),
    )


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="config")
def main(cfg: DictConfig):
    """Main function decorated with Hydra.

    Args:
        cfg: Hydra configuration object
    """
    # Convert OmegaConf to regular dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)

    # Get algorithm from config (default to ppo)
    algorithm = config.get("ALGORITHM", "ppo").lower()

    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )

    spec = ALGORITHM_REGISTRY[algorithm]

    rng = jax.random.PRNGKey(config.get("SEED", 30))

    # Create run directory
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = spec.run_dir(config, runs_root, ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {run_dir}")

    update_latest_symlink(run_dir, runs_root)

    # Setup tensorboard
    writer = SummaryWriter(f"{run_dir}")

    # Setup optional Weights & Biases logging
    wandb_run = init_wandb(config, run_dir, algorithm)

    # Create unified training function
    rng, init_rng = jax.random.split(rng)
    train_fn = make_train(
        config,
        spec,
        writer,
        init_rng,
        run_dir,
        wandb_run=wandb_run,
    )

    # JIT compile and train
    train_jit = jax.jit(train_fn)
    print("Compiling and training...")
    start_time = time.time()
    final_state = train_jit(rng)
    jax.block_until_ready(final_state)
    end_time = time.time()

    # Log training speed (total time includes compilation)
    total_time = end_time - start_time
    total_steps = config["TOTAL_TIMESTEPS"]

    writer.add_scalar("perf/total_time_seconds", total_time, total_steps)

    # Save params
    params = final_state.train_state.params
    pure_params = params.to_pure_dict()
    (run_dir / "policy_params.msgpack").write_bytes(serialization.to_bytes(pure_params))

    # Add environment-specific attributes to config
    env = ENV_REGISTRY[config["ENV_NAME"]]()
    env_params = env.default_params
    config["env"] = {}

    # List of attributes to extract from env and env_params
    env_attrs = [
        ("action_dim", "env", "action_dim"),
        ("obs_dim", "env", "obs_dim"),
        ("param_dim", "env", "param_dim"),
        ("MIN_ACTION", "env_params", "min_action"),
        ("MAX_ACTION", "env_params", "max_action"),
        ("MIN_INPUT", "env_params", "min_input"),
        ("MAX_INPUT", "env_params", "max_input"),
    ]

    for config_key, source, attr_name in env_attrs:
        try:
            source_obj = env if source == "env" else env_params
            attr_value = getattr(source_obj, attr_name)
            # Convert to int if it's a scalar, to list if it's an array
            if isinstance(attr_value, (int, float, bool)):
                config["env"][config_key] = (
                    int(attr_value)
                    if isinstance(attr_value, (int, bool))
                    else attr_value
                )
            else:
                config["env"][config_key] = attr_value.tolist()
        except AttributeError:
            print(
                f"Warning: Environment attribute '{attr_name}' not found in {source}. Skipping."
            )

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Log hparams
    hparam_dict = {
        k: v for k, v in config.items() if isinstance(v, (int, float, bool, str))
    }
    writer.add_hparams(hparam_dict, {}, run_name=".")

    writer.close()

    print(f"Training complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
