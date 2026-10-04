"""Base classes and shared utilities for PPO-family algorithms."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from buffers import RunnerState


# ---------------------------------------------------------------------------
# Default implementations for common algorithm patterns.
# Algorithms only override the ones where they differ.
# ---------------------------------------------------------------------------


def default_extract_losses(loss_info):
    """Standard 7-metric loss extraction used by most PPO-family algorithms."""
    total_losses, (
        value_losses,
        actor_losses,
        entropies,
        old_approx_kl,
        approx_kl,
        clipfracs,
    ) = loss_info
    losses = {
        "total": total_losses.mean(),
        "value": value_losses.mean(),
        "actor": actor_losses.mean(),
        "entropy": entropies.mean(),
        "old_approx_kl": old_approx_kl.mean(),
        "approx_kl": approx_kl.mean(),
        "clipfrac": jnp.concatenate(clipfracs).mean(),
    }
    return losses, None


def default_init_runner_state(config, env, env_params, networks, rng):
    """Standard runner state init: reset env, no extras."""
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)
    return RunnerState(
        train_state=networks["train_state"],
        env_state=env_state,
        last_obs=obsv,
        rng=_rng,
    )


def default_run_dir(algo_name, config, runs_root, ts):
    """Standard run directory: ``<env>/<ALGO>_[DR_]<timestamp>``."""
    suffix = f"{algo_name}_DR_{ts}" if config.get("DOMAIN_RANDOMIZATION", False) else f"{algo_name}_{ts}"
    return runs_root / config["ENV_NAME"] / suffix


# ---------------------------------------------------------------------------
# AlgorithmSpec
# ---------------------------------------------------------------------------
def _policy_dist_from_model(model, obs, rng):
    """Return policy distribution while supporting model signatures with/without rng."""
    if isinstance(obs, dict) and "local_obs" in obs:
        obs = obs["local_obs"]
    try:
        model_out = model(obs, rng)
    except TypeError:
        model_out = model(obs)
    return model_out[0] if isinstance(model_out, tuple) else model_out

def default_make_eval_step(config, graphdef, env, env_params, rngs_state=None, **kwargs):
    """Standard evaluation step for single-model algorithms (PPO, CPPO)."""
    if rngs_state is None:
        rngs_state = {}

    def _eval_step(eval_state: RunnerState, _unused):
        rng, model_rng = jax.random.split(eval_state.rng)
        model = nnx.merge(graphdef, eval_state.train_state.params, rngs_state)
        pi = _policy_dist_from_model(model, eval_state.last_obs, model_rng)
        action = jnp.clip(pi.mode(), -1.0, 1.0)

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

@dataclass
class AlgorithmSpec:
    """Specification for a PPO-family algorithm variant.

    Only ``algo_name`` and the six "required" callables must be provided.
    Everything else has sensible defaults that work for the standard PPO
    training loop.

    Required:
        algo_name:       Short uppercase label (e.g. ``"PPO"``, ``"SPPO_AUG"``).
        wrap_env:        Function to apply algorithm-specific wrappers.
        make_loss_fn:    Factory that returns the loss function.
        make_collect_fn: Factory for trajectory collection step.
        make_update_fn:  Factory for update epoch step.
        calculate_gae:   GAE computation returning a results dict.
        init_networks:   Network / TrainState initialisation.

    Optional (defaults provided):
        extract_losses:       Extract logging metrics from loss_info.
        init_runner_state:    Create the initial RunnerState.
        run_dir:              Build the run directory path.
        make_eval_step:       Factory for evaluation step.
    """

    # --- required ---
    algo_name: str
    wrap_env: Callable
    make_loss_fn: Callable
    make_collect_fn: Callable
    make_update_fn: Callable
    calculate_gae: Callable
    init_networks: Callable

    # --- optional (have defaults) ---
    extract_losses: Optional[Callable] = None
    init_runner_state: Optional[Callable] = None
    run_dir: Optional[Callable] = None
    make_eval_step: Optional[Callable] = None

    def __post_init__(self):
        if self.extract_losses is None:
            self.extract_losses = default_extract_losses
        if self.init_runner_state is None:
            self.init_runner_state = default_init_runner_state
        if self.make_eval_step is None:
            self.make_eval_step = default_make_eval_step
        if self.run_dir is None:
            name = self.algo_name
            self.run_dir = lambda config, runs_root, ts: default_run_dir(
                name, config, runs_root, ts
            )

