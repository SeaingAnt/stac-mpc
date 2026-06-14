"""PPO algorithm specification."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState

from .base import AlgorithmSpec
from .common import gae_standard
from net_utils import create_optimizer, make_linear_schedule
from buffers import Transition, RunnerState, UpdateState
from networks import ActorCritic
from env.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    DomainRandomizationWrapper,
    ClipAction,
    ResetEnvWrapper,
)


def wrap_env(env, config):
    """Apply PPO-specific environment wrappers."""
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)  # Don't remove
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    return env


def make_loss_fn(config, graphdef, rngs_state=None, state_template=None, **kwargs):
    """Create PPO loss function.

    Returns a function with signature:
        loss_fn(params, traj_batch, advantages, targets) -> (total_loss, aux_metrics)
    """

    def _loss_fn(params, traj_batch, advantages, targets):
        # RERUN NETWORK
        model = nnx.merge(graphdef, params)
        pi, value = model(traj_batch.obs)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)

        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfracs = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])

        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        )
        return total_loss, (
            value_loss,
            loss_actor,
            entropy,
            old_approx_kl,
            approx_kl,
            clipfracs,
        )

    return _loss_fn


def make_collect_fn(config, env, env_params, networks):
    """Create trajectory collection function for PPO.

    Args:
        config: Training configuration
        env: Wrapped environment
        env_params: Environment parameters
        networks: Dict returned by init_networks containing 'network', 'train_state', etc.

    Returns a function for use with jax.lax.scan that collects transitions.
    """
    graphdef = networks["graphdef"]

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        model = nnx.merge(graphdef, runner_state.train_state.params)
        pi, value = model(runner_state.last_obs)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )
        
        if "real_next_obs" in info:
            _, real_next_value = model(info["real_next_obs"])
            info["real_next_value"] = real_next_value

        obsv = jax.lax.stop_gradient(obsv)
        env_state = jax.lax.stop_gradient(env_state)
        transition = Transition(
            done, action, value, reward, log_prob, runner_state.last_obs, info
        )
        new_runner_state = RunnerState(
            train_state=runner_state.train_state,
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
        )
        return new_runner_state, transition

    return _env_step


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    """Create update epoch function for PPO.

    Args:
        config: Training configuration
        loss_fn: Loss function created by make_loss_fn
        shuffle_batch_fn: Function to shuffle and batch trajectories
        networks: Dict returned by init_networks (unused in PPO but needed for interface)

    Returns a function for use with jax.lax.scan that performs one epoch of updates.
    """

    def _update_epoch(update_state: UpdateState, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        rng, _rng = jax.random.split(update_state.rng)
        minibatches = shuffle_batch_fn(
            _rng,
            (update_state.traj_batch, update_state.advantages, update_state.targets),
            config,
        )
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, update_state.train_state, minibatches
        )
        new_update_state = UpdateState(
            train_state=train_state,
            traj_batch=update_state.traj_batch,
            advantages=update_state.advantages,
            targets=update_state.targets,
            rng=rng,
            extras=update_state.extras,
        )
        return new_update_state, total_loss

    return _update_epoch


def calculate_gae(config, traj_batch, networks, collect_state):
    """Calculate GAE for PPO."""
    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    _, last_val = model(collect_state.last_obs)
    advantages, targets = gae_standard(config, traj_batch, last_val)
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Initialize network and train state for PPO.

    Returns:
        dict with keys: graphdef, train_state, and any secondary networks/states
    """
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).shape[0]

    network = ActorCritic(
        obs_dim,
        action_dim,
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        rngs=nnx.Rngs(int(rng[0])),
    )

    graphdef, state = nnx.split(network)

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=None,
        params=state,
        tx=tx,
    )

    return {
        "graphdef": graphdef,
        "train_state": train_state,
    }


SPEC = AlgorithmSpec(
    algo_name="PPO",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
)
