"""MAPPO algorithm specification."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState

from .base import AlgorithmSpec
from algorithms.common import gae_standard
from net_utils import create_optimizer, make_linear_schedule
from buffers import Transition, RunnerState, UpdateState
from networks import MAPPOActor, MAPPOCritic
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
    """Apply MAPPO-specific environment wrappers."""
    if not env.decentralized:
        raise ValueError(
            "MAPPO requires decentralized environments with separate local_obs and global_state."
        )
    env = GymnaxWrapper(env)
    env = ClipAction(env)
    if config.get("DOMAIN_RANDOMIZATION", False):
        env = DomainRandomizationWrapper(env, env.default_params)
    else:
        env = ResetEnvWrapper(env)
    env = LogWrapper(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    return env


def make_loss_fn(config, graphdef, **kwargs):
    """Create MAPPO loss function."""

    def _loss_fn(params, traj_batch, advantages, targets):
        actor_model = nnx.merge(graphdef["actor"], params["actor"])
        critic_model = nnx.merge(graphdef["critic"], params["critic"])
        # obs is a dictionary with "local_obs" and "global_state"
        pi = actor_model(traj_batch.obs["local_obs"])
        value = critic_model(traj_batch.obs["global_state"])
        log_prob = pi.log_prob(
            traj_batch.action
        )  # Action shape: (batch, num_agents, act_dim)

        # CALCULATE VALUE LOSS (Critic is centralized, evaluates global state)
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        # Use individual agent ratios with the shared centralized advantage
        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)

        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfracs = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])

        # Broadcast gae (batch,) to (batch, num_agents) for per-agent ratios
        gae = jnp.expand_dims(advantages, axis=-1)

        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        # Entropy usually averaged across agents
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
    """Create trajectory collection function for MAPPO."""
    graphdef = networks["graphdef"]

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)

        actor_model = nnx.merge(
            graphdef["actor"], runner_state.train_state.params["actor"]
        )
        critic_model = nnx.merge(
            graphdef["critic"], runner_state.train_state.params["critic"]
        )

        pi = actor_model(runner_state.last_obs["local_obs"])
        value = critic_model(runner_state.last_obs["global_state"])
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )

        if "real_next_obs" in info:
            real_next_value = critic_model(info["real_next_obs"]["global_state"])
            info["real_next_value"] = real_next_value

        obsv = jax.tree_util.tree_map(jax.lax.stop_gradient, obsv)
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
    """Create update epoch function for MAPPO."""

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
    """Calculate GAE for MAPPO."""
    critic_model = nnx.merge(
        networks["graphdef"]["critic"], collect_state.train_state.params["critic"]
    )
    last_val = critic_model(collect_state.last_obs["global_state"])
    advantages, targets = gae_standard(config, traj_batch, last_val)
    
    # Normalize advantages globally across the entire trajectory batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


def init_networks(config, env, env_params, rng, load_params_fn=None):
    """Initialize network and train state for MAPPO."""
    # Assuming environment provides a dictionary for observation space
    obs_space = env.observation_space(env_params)
    local_obs_dim = obs_space.spaces["local_obs"].shape[-1]
    global_state_dim = obs_space.spaces["global_state"].shape[-1]

    # Assuming action space is (num_agents, action_dim_per_agent)
    action_space = env.action_space(env_params)
    num_agents = action_space.shape[0]
    action_dim_per_agent = action_space.shape[1]

    actor_network = MAPPOActor(
        local_obs_dim=local_obs_dim,
        action_dim_per_agent=action_dim_per_agent,
        num_agents=num_agents,
        activation=config["ACTIVATION"],
        actor_layer_sizes=tuple(config.get("ACTOR_LAYER_SIZES", (256, 256))),
        rngs=nnx.Rngs(int(rng[0])),
    )

    critic_network = MAPPOCritic(
        global_state_dim=global_state_dim,
        activation=config["ACTIVATION"],
        critic_layer_sizes=tuple(config.get("CRITIC_LAYER_SIZES", (256, 256))),
        rngs=nnx.Rngs(int(rng[0]) + 1),
    )

    actor_graphdef, actor_state = nnx.split(actor_network)
    critic_graphdef, critic_state = nnx.split(critic_network)

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)

    # To interoperate seamlessly with the RunnerState / UpdateState which expect
    # a single TrainState, we package actor and critic params in one TrainState
    train_state = TrainState.create(
        apply_fn=None,
        params={"actor": actor_state, "critic": critic_state},
        tx=tx,
    )

    graphdef = {"actor": actor_graphdef, "critic": critic_graphdef}

    return {
        "graphdef": graphdef,
        "train_state": train_state,
    }


def make_eval_step(config, graphdef, env, env_params):
    """Create deterministic env-step function for MAPPO evaluation rollouts."""

    def _eval_step(eval_state: RunnerState, _unused):
        rng, model_rng = jax.random.split(eval_state.rng)

        # In MAPPO, graphdef and params are dictionaries containing 'actor' and 'critic'
        actor_model = nnx.merge(
            graphdef["actor"], eval_state.train_state.params["actor"]
        )

        # Policy depends only on local observations
        pi = actor_model(eval_state.last_obs["local_obs"])
        # Clip the mode of the distribution to stay within [-1, 1] action bounds
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


SPEC = AlgorithmSpec(
    algo_name="MAPPO",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    init_networks=init_networks,
    make_eval_step=make_eval_step,
)
