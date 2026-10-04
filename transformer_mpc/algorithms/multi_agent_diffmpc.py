""" "DiffMPC Algorithm implementation."""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.train_state import TrainState
from types import SimpleNamespace
from functools import partial

from .base import AlgorithmSpec
from .common import gae_standard
from net_utils import create_optimizer, make_linear_schedule
from buffers import Transition, RunnerState, UpdateState
from networks import MAPPODiffMPCActor, MAPPOCritic
from env.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    DomainRandomizationWrapper,
    ClipAction,
    ResetEnvWrapper,
)
from env.math import skew, quat2rotm, quat_product

import mpx.utils.mpc_wrapper as base_mpc_wrapper


def _safe_control_violation(u, min_input, max_input):
    lower = jax.nn.softplus(10.0 * (min_input - u)) / 10.0
    upper = jax.nn.softplus(10.0 * (u - max_input)) / 10.0
    return lower + upper


def make_mpx_solve_fn(env_dynamics, env_params, horizon, dt, nx, nu, pcg_iters=50):
    def dynamics(x, u, t, parameter):
        del t, parameter
        dx = env_dynamics.state_dot(x, u, None)
        x_mid = x + (dt / 2.0) * dx
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        return x_next

    def cost(W, reference, x, u, t):
        weights = W
        half_dim = horizon * (nx + nu)

        Q_R_traj = jnp.clip(jax.nn.softplus(weights[:half_dim]) * 10.0, 0.0, 100.0)
        p_logits = weights[half_dim:]

        Q_R_traj = Q_R_traj.reshape((horizon, nx + nu))
        Q_traj = Q_R_traj[:, :nx] + 0.05
        R_traj = Q_R_traj[:, nx:] + 0.05

        p_mapped = 10.0 * jnp.tanh(p_logits)
        P_traj = p_mapped.reshape((horizon, nx + nu))

        idx = jnp.minimum(t, horizon - 1)
        Qt = Q_traj[idx]
        Rt = R_traj[idx]
        Pt = P_traj[idx]

        P_x = Pt[:nx]
        P_u = Pt[nx:]

        state_error = x - reference

        stage_cost = (
            0.5 * (jnp.sum(Qt * state_error**2) + jnp.sum(Rt * u**2))
            + jnp.sum(P_x * state_error)
            + jnp.sum(P_u * u)
        )
        term_cost = 0.5 * jnp.sum(Qt * state_error**2) + jnp.sum(P_x * state_error)

        return jnp.where(t == horizon, term_cost, stage_cost)

    config = SimpleNamespace(
        solver_mode="primal_dual",
        cost=cost,
        dynamics=dynamics,
        hessian_approx=None,
    )

    _, solve_fn = base_mpc_wrapper.build_solver_step(
        config=config,
        cost=cost,
        dynamics=dynamics,
        hessian_approx=None,
        limited_memory=False,
    )
    return solve_fn


def solve_mpc(weights, physical_state, reference, env_params, env_dynamics, solve_fn, horizon, dt, nx, nu, mpc_iters=1):
    nominal_hover = jnp.zeros(nu)

    W = weights
    parameter = jnp.zeros(1)

    init_U0 = jnp.tile(nominal_hover, (horizon, 1))
    
    def rollout_step(x, u):
        dx = env_dynamics.state_dot(x, u, None)
        x_mid = x + (dt / 2.0) * dx
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        return x_next, x_next

    _, X_traj = jax.lax.scan(rollout_step, physical_state, init_U0)
    init_X0 = jnp.concatenate([physical_state[None, :], X_traj], axis=0)
    
    init_V0 = jnp.zeros((horizon + 1, nx))

    def solver_step(carry, _):
        X, U, V = carry
        X_next, U_next, V_next = solve_fn(
            reference, parameter, W, physical_state, X, U, V
        )
        return (X_next, U_next, V_next), None

    (sol_X, sol_U, sol_V), _ = jax.lax.scan(
        solver_step, (init_X0, init_U0, init_V0), None, length=mpc_iters
    )

    physical_action = sol_U[0]

    physical_action = jnp.where(
        jnp.isnan(physical_action).any(), nominal_hover, physical_action
    )

    physical_action = jnp.clip(
        physical_action, -1.0, 1.0
    )

    return physical_action


def wrap_env(env, config):
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


def make_collect_fn(config, env, env_params, networks):
    graphdef = networks["graphdef"]

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
            return None, None
        return base.pos, base.target

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        actor_model = nnx.merge(graphdef["actor"], runner_state.train_state.params["actor"])
        critic_model = nnx.merge(graphdef["critic"], runner_state.train_state.params["critic"])

        physical_state, reference = get_physical_state(runner_state.env_state)
        pi = actor_model(runner_state.last_obs["local_obs"], physical_state=physical_state, reference=reference)
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

        info["physical_state"] = physical_state
        info["reference"] = reference

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


def make_loss_fn(config, graphdef, *args, **kwargs):
    def _loss_fn(params, traj_batch, advantages, targets):
        actor_model = nnx.merge(graphdef["actor"], params["actor"])
        critic_model = nnx.merge(graphdef["critic"], params["critic"])

        physical_state = traj_batch.info["physical_state"]
        reference = traj_batch.info["reference"]
        pi = actor_model(traj_batch.obs["local_obs"], physical_state=physical_state, reference=reference)
        value = critic_model(traj_batch.obs["global_state"])

        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        log_ratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(log_ratio)

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


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
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
    critic_model = nnx.merge(networks["graphdef"]["critic"], collect_state.train_state.params["critic"])
    last_val = critic_model(collect_state.last_obs["global_state"])
    advantages, targets = gae_standard(config, traj_batch, last_val)
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


def extract_losses(loss_info):
    total_loss, aux = loss_info
    value_loss, loss_actor, entropy, old_approx_kl, approx_kl, clipfracs = aux

    losses = {
        "total": total_loss.mean(),
        "value": value_loss.mean(),
        "actor": loss_actor.mean(),
        "entropy": entropy.mean(),
        "old_approx_kl": old_approx_kl.mean(),
        "approx_kl": approx_kl.mean(),
        "clipfrac": clipfracs.mean(),
    }
    return losses, {}


def init_networks(config, env, env_params, rng, load_params_fn=None):
    obs_space = env.observation_space(env_params)
    local_obs_dim = obs_space.spaces["local_obs"].shape[-1]
    global_state_dim = obs_space.spaces["global_state"].shape[-1]
    
    num_agents = env.num_agents
    
    # Each MPC is 2D
    nx = 2
    nu = 2
    
    horizon = getattr(env_params, "mpc_horizon", config.get("MPC_HORIZON", 10))

    mpc_weight_dim = 2 * horizon * (nx + nu)
    env_action_dim_per_agent = 2

    env_dynamics = env.mpc_dynamics(nx, nu, env_params)
    solve_fn = make_mpx_solve_fn(
        env_dynamics=env_dynamics,
        env_params=env_params,
        horizon=horizon,
        dt=getattr(env_params, "dt", 0.1),
        nx=nx,
        nu=nu,
        pcg_iters=config.get("MPC_PCG_ITERS", 50),
    )

    def mpc_layer(weights, physical_state, reference):
        return solve_mpc(
            weights,
            physical_state,
            reference,
            env_params,
            env_dynamics,
            solve_fn,
            horizon,
            getattr(env_params, "dt", 0.1),
            nx,
            nu,
            mpc_iters=config.get("MPC_ITERS", 1),
        )

    @jax.custom_vjp
    def safe_mpc_layer(weights, physical_state, reference):
        return mpc_layer(weights, physical_state, reference)

    def safe_mpc_layer_fwd(weights, physical_state, reference):
        return mpc_layer(weights, physical_state, reference), (weights, physical_state, reference)

    def safe_mpc_layer_bwd(res, g):
        weights, physical_state, reference = res

        g = jnp.clip(g, -10.0, 10.0)
        g = jnp.where(jnp.abs(g) < 1e-10, 1e-10 * jnp.sign(g + 1e-20), g)

        _, vjp_fn = jax.vjp(mpc_layer, weights, physical_state, reference)
        g_w, g_p, g_r = vjp_fn(g)

        g_w = jax.tree_util.tree_map(lambda x: jnp.clip(jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0), g_w)
        g_p = jax.tree_util.tree_map(lambda x: (jnp.clip(jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0) if x is not None else None), g_p)
        g_r = jax.tree_util.tree_map(lambda x: (jnp.clip(jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0) if x is not None else None), g_r)

        return g_w, g_p, g_r

    safe_mpc_layer.defvjp(safe_mpc_layer_fwd, safe_mpc_layer_bwd)

    # Vmap over agents! (weights, pos, target are per-agent)
    vmap_mpc_layer_agents = jax.vmap(safe_mpc_layer, in_axes=(0, 0, 0))
    # Vmap over batch
    vmap_mpc_layer_batch = jax.vmap(vmap_mpc_layer_agents, in_axes=(0, 0, 0))

    actor_network = MAPPODiffMPCActor(
        local_obs_dim=local_obs_dim,
        mpc_weight_dim=mpc_weight_dim,
        env_action_dim_per_agent=env_action_dim_per_agent,
        num_agents=num_agents,
        mpc_fn=vmap_mpc_layer_batch,
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

    train_state = TrainState.create(
        apply_fn=None,
        params={"actor": actor_state, "critic": critic_state},
        tx=tx,
    )

    graphdef = {"actor": actor_graphdef, "critic": critic_graphdef}

    if load_params_fn is not None:
        train_state = train_state.replace(params=load_params_fn(train_state.params))

    return {
        "graphdef": graphdef,
        "train_state": train_state,
    }

def make_eval_step(config, graphdef, env, env_params, **kwargs):
    """Create deterministic env-step function for DIFFMPC evaluation rollouts."""

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
            return None, None
        return base.pos, base.target

    def _eval_step(eval_state: RunnerState, _unused):
        rng, model_rng = jax.random.split(eval_state.rng)
        actor_model = nnx.merge(graphdef["actor"], eval_state.train_state.params["actor"])
        
        physical_state, reference = get_physical_state(eval_state.env_state)
        
        pi = actor_model(eval_state.last_obs["local_obs"], physical_state=physical_state, reference=reference)
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
    algo_name="DIFFMPC",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    extract_losses=extract_losses,
    init_networks=init_networks,
    make_eval_step=make_eval_step,
)
