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
        x_mid = x_mid.at[3:7].set(x_mid[3:7] / (jnp.linalg.norm(x_mid[3:7]) + 1e-12))
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        x_next = x_next.at[3:7].set(x_next[3:7] / (jnp.linalg.norm(x_next[3:7]) + 1e-12))
        return x_next

    def cost(W, reference, x, u, t):
        del reference
        weights = W
        half_dim = horizon * (nx + nu)

        # Softplus preserves gradients everywhere but cleanly bounds from below.
        # This prevents both the catastrophic vanishing gradients from saturated tanh
        # and the extreme Q matrix explosion from raw clipping.
        Q_R_traj = jnp.clip(jax.nn.softplus(weights[:half_dim]) * 10.0, 0.0, 100.0)
        # Q_R_traj = jax.nn.tanh(weights[:half_dim]) ** 2 * 100.0
        p_logits = weights[half_dim:]

        Q_R_traj = Q_R_traj.reshape((horizon, nx + nu))
        # Keep the +0.05 minimum floor for strict positive definiteness
        Q_traj = Q_R_traj[:, :nx] + 0.05
        R_traj = Q_R_traj[:, nx:] + 0.05

        # P can naturally be negative, so we map it symmetrically
        p_mapped = 10.0 * jnp.tanh(p_logits)
        P_traj = p_mapped.reshape((horizon, nx + nu))

        idx = jnp.minimum(t, horizon - 1)
        Qt = Q_traj[idx]
        Rt = R_traj[idx]
        Pt = P_traj[idx]

        P_x = Pt[:nx]
        P_u = Pt[nx:]

        control_violation = _safe_control_violation(
            u,
            env_params.min_input,
            env_params.max_input,
        )
        control_constraint_cost = 25.0 * jnp.sum(control_violation**2)

        stage_cost = (
            0.5 * (jnp.sum(Qt * x**2) + jnp.sum(Rt * u**2))
            + jnp.sum(P_x * x)
            + jnp.sum(P_u * u)
            + control_constraint_cost
        )
        term_cost = 0.5 * jnp.sum(Qt * x**2) + jnp.sum(P_x * x)

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


def solve_mpc(weights, physical_state, env_params, env_dynamics, solve_fn, horizon, dt, nx, nu, mpc_iters=1):
    nominal_hover = jnp.array([env_params.m * env_params.g, 0.0, 0.0, 0.0])

    W = weights
    reference = jnp.zeros(1)
    parameter = jnp.zeros(1)

    init_U0 = jnp.tile(nominal_hover, (horizon, 1))
    
    def rollout_step(x, u):
        dx = env_dynamics.state_dot(x, u, None)
        x_mid = x + (dt / 2.0) * dx
        x_mid = x_mid.at[3:7].set(x_mid[3:7] / (jnp.linalg.norm(x_mid[3:7]) + 1e-12))
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        x_next = x_next.at[3:7].set(x_next[3:7] / (jnp.linalg.norm(x_next[3:7]) + 1e-12))
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
        physical_action, env_params.min_input, env_params.max_input
    )

    input_span = env_params.max_input - env_params.min_input
    action_span = env_params.max_action - env_params.min_action
    normalized_action = (
        env_params.min_action
        + ((physical_action - env_params.min_input) / input_span) * action_span
    )

    return normalized_action


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
        return jnp.concatenate([base.pos, base.attitude, base.vel, base.omega], axis=-1)

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        model = nnx.merge(graphdef, runner_state.train_state.params)

        physical_state = get_physical_state(runner_state.env_state)
        pi, value = model(runner_state.last_obs, physical_state=physical_state)

        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state, reward, done, info = env.step(
            rng_step, runner_state.env_state, action, env_params
        )

        if "real_next_obs" in info:
            real_next_value = model.critic(info["real_next_obs"])
            info["real_next_value"] = real_next_value

        info["physical_state"] = physical_state

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
        model = nnx.merge(graphdef, params)

        physical_state = traj_batch.info["physical_state"]
        pi, value = model(traj_batch.obs, physical_state=physical_state)

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
    model = nnx.merge(networks["graphdef"], collect_state.train_state.params)
    last_val = model.critic(collect_state.last_obs)
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
    obs_dim = env.observation_space(env_params).shape[0]

    nx = env.state_dim if hasattr(env, "state_dim") else 13
    nu = env.num_actions if hasattr(env, "num_actions") else 4
    # Prefer MPC horizon from environment parameters; fall back to config
    horizon = getattr(env_params, "mpc_horizon", config.get("MPC_HORIZON", 10))

    mpc_weight_dim = 2 * horizon * (nx + nu)
    env_action_dim = 4

    env_dynamics = env.mpc_dynamics(nx, nu, env_params)
    solve_fn = make_mpx_solve_fn(
        env_dynamics=env_dynamics,
        env_params=env_params,
        horizon=horizon,
        dt=getattr(env_params, "dt", 0.02),
        nx=nx,
        nu=nu,
        pcg_iters=config.get("MPC_PCG_ITERS", 50),
    )

    def mpc_layer(weights, physical_state):
        return solve_mpc(
            weights,
            physical_state,
            env_params,
            env_dynamics,
            solve_fn,
            horizon,
            getattr(env_params, "dt", 0.02),
            nx,
            nu,
            mpc_iters=config.get("MPC_ITERS", 1),
        )

    @jax.custom_vjp
    def safe_mpc_layer(weights, physical_state):
        return mpc_layer(weights, physical_state)

    def safe_mpc_layer_fwd(weights, physical_state):
        return mpc_layer(weights, physical_state), (weights, physical_state)

    def safe_mpc_layer_bwd(res, g):
        weights, physical_state = res

        # Keep the MPC adjoint bounded so gradients remain numerically stable.

        g = jnp.clip(g, -10.0, 10.0)
        g = jnp.where(jnp.abs(g) < 1e-10, 1e-10 * jnp.sign(g + 1e-20), g)

        _, vjp_fn = jax.vjp(mpc_layer, weights, physical_state)
        g_w, g_p = vjp_fn(g)

        g_w = jax.tree_util.tree_map(
            lambda x: jnp.clip(
                jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -10.0, 10.0
            ),
            g_w,
        )
        g_p = jax.tree_util.tree_map(
            lambda x: (
                jnp.clip(jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -10.0, 10.0)
                if x is not None
                else None
            ),
            g_p,
        )

        return g_w, g_p

    safe_mpc_layer.defvjp(safe_mpc_layer_fwd, safe_mpc_layer_bwd)

    vmap_mpc_layer = jax.vmap(safe_mpc_layer, in_axes=(0, 0))

    network = ActorCritic(
        obs_dim=obs_dim,
        action_dim=mpc_weight_dim,
        env_action_dim=env_action_dim,
        mpc_fn=vmap_mpc_layer,
        activation=config["ACTIVATION"],
        rngs=nnx.Rngs(rng),
    )
    graphdef, params = nnx.split(network)

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)
    train_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=tx,
    )

    if load_params_fn is not None:
        train_state = train_state.replace(params=load_params_fn(train_state.params))

    return {
        "network": network,
        "graphdef": graphdef,
        "train_state": train_state,
    }


SPEC = AlgorithmSpec(
    algo_name="DIFFMPC",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    extract_losses=extract_losses,
    init_networks=init_networks,
)
