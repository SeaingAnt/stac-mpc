""" "Multi-Agent DiffMPC Algorithm with Transformer architecture and stability regularization.

Adapts the contraction-metric stability condition from diffmpc_transformer_stab.py
for the simpler 2D single-integrator multi-agent navigation system.
"""

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
from networks import MAPPOTransformerDiffMPCActor, MAPPOCritic
from env.wrappers import (
    GymnaxWrapper,
    LogWrapper,
    VecEnv,
    NormalizeVecReward,
    DomainRandomizationWrapper,
    ClipAction,
    ResetEnvWrapper,
)

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
        Q_traj = Q_R_traj[:, :nx] + 0.1
        R_traj = Q_R_traj[:, nx:] + 0.1

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


def solve_mpc(
    weights,
    physical_state,
    reference,
    env_params,
    env_dynamics,
    solve_fn,
    horizon,
    dt,
    nx,
    nu,
    mpc_iters=1,
    return_trajectory=False,
):
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

    physical_action = jnp.clip(physical_action, -1.0, 1.0)

    if return_trajectory:
        return physical_action, sol_X, sol_U
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
        rngs_state = networks.get("rngs_state", {})
        actor_rngs = rngs_state.get("actor", {})
        critic_rngs = rngs_state.get("critic", {})
        actor_model = nnx.merge(
            graphdef["actor"],
            runner_state.train_state.params["actor"],
            actor_rngs,
        )
        critic_model = nnx.merge(
            graphdef["critic"],
            runner_state.train_state.params["critic"],
            critic_rngs,
        )

        physical_state, reference = get_physical_state(runner_state.env_state)
        pi = actor_model(
            runner_state.last_obs["local_obs"],
            physical_state=physical_state,
            reference=reference,
        )
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


# ---------------------------------------------------------------------------
# Contraction metric computation (Discrete Riccati)
# ---------------------------------------------------------------------------

def compute_contraction_metric_pure_jax(
    A_seq, B_seq, Q_seq, R_seq, tol=1e-5, max_iters=50
):
    """Compute the contraction metric M_0 via a backward Riccati recursion.

    For the multi-agent 2D nav system (single integrator), the linearized
    dynamics are A=I, B=v_max*dt*I at every step, so this is straightforward.
    """
    A_T = A_seq[-1]
    B_T = B_seq[-1]
    Q_T = Q_seq[-1]
    R_T = R_seq[-1]

    # Use jax.lax.scan instead of while_loop to support reverse-mode autodiff.
    def dare_step(carry, _):
        M_curr = carry
        R_tilde = R_T + B_T.T @ M_curr @ B_T
        cross_term = B_T.T @ M_curr @ A_T
        K_gain = jax.scipy.linalg.solve(R_tilde, cross_term, assume_a="pos")
        M_next = A_T.T @ M_curr @ A_T - (A_T.T @ M_curr @ B_T) @ K_gain + Q_T
        return M_next, None

    M_T, _ = jax.lax.scan(dare_step, Q_T, None, length=max_iters)

    def dre_step(M_next, inputs):
        A_i, B_i, Q_i, R_i = inputs
        R_tilde = R_i + B_i.T @ M_next @ B_i
        cross_term = B_i.T @ M_next @ A_i
        K_i = jax.scipy.linalg.solve(R_tilde, cross_term, assume_a="pos")
        M_curr = A_i.T @ M_next @ A_i - (A_i.T @ M_next @ B_i) @ K_i + Q_i
        return M_curr, None

    inputs_reversed = (
        A_seq[:-1][::-1],
        B_seq[:-1][::-1],
        Q_seq[:-1][::-1],
        R_seq[:-1][::-1],
    )

    M_0, _ = jax.lax.scan(dre_step, M_T, inputs_reversed)
    return M_0


# ---------------------------------------------------------------------------
# Loss function with stability regularization
# ---------------------------------------------------------------------------

def make_loss_fn(config, graphdef, rngs_state=None, state_template=None, **kwargs):
    if rngs_state is None:
        rngs_state = {}
    actor_rngs = rngs_state.get("actor", {})
    critic_rngs = rngs_state.get("critic", {})

    env = kwargs.get("env")
    env_params = kwargs.get("env_params")

    # Per-agent state/action dims for the 2D nav system
    nx = 2
    nu = 2
    horizon = getattr(env_params, "mpc_horizon", config.get("MPC_HORIZON", 10))
    dt = getattr(env_params, "dt", 0.1) if env_params else 0.1

    env_dynamics = env.mpc_dynamics(nx, nu, env_params) if env else None

    if env and env_params:
        solve_fn = make_mpx_solve_fn(
            env_dynamics=env_dynamics,
            env_params=env_params,
            horizon=horizon,
            dt=dt,
            nx=nx,
            nu=nu,
            pcg_iters=config.get("MPC_PCG_ITERS", 50),
        )
    else:
        solve_fn = None

    def _loss_fn(params, traj_batch, advantages, targets):
        actor_model = nnx.merge(graphdef["actor"], params["actor"], actor_rngs)
        critic_model = nnx.merge(graphdef["critic"], params["critic"], critic_rngs)

        physical_state = traj_batch.info["physical_state"]
        reference = traj_batch.info["reference"]
        pi = actor_model(
            traj_batch.obs["local_obs"],
            physical_state=physical_state,
            reference=reference,
        )
        value = critic_model(traj_batch.obs["global_state"])

        log_prob = pi.log_prob(traj_batch.action)

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        log_ratio = log_prob - traj_batch.log_prob

        # Prevent extreme PPO policy loss spikes / gradient explosions
        log_ratio = jnp.clip(log_ratio, -5.0, 5.0)

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

        # ------------------------------------------------------------------
        # Transformer Lipschitz regularization + contraction stability
        # ------------------------------------------------------------------
        def soft_max(x, alpha=5000.0, axis=None):
            """Numerically stable soft maximum."""
            return jax.nn.logsumexp(alpha * x, axis=axis) / alpha

        def soft_matrix_inf_norm(W, alpha=5000.0):
            """Soft induced infinity norm for a matrix."""
            row_sums = jnp.sum(jnp.abs(W), axis=-1)
            return soft_max(row_sums, alpha=alpha, axis=0)

        reg_loss = 0.0
        if hasattr(actor_model, "encoder_blocks"):
            epsilon = 0.1
            for block in actor_model.encoder_blocks:
                gamma1_inf = soft_max(jnp.abs(block.mha_ln.scale.value))
                gamma2_inf = soft_max(jnp.abs(block.ffn_ln.scale.value))

                head_term = 0.0
                sum_w_v = 0.0
                D = actor_model.obs_seq_len / jnp.sqrt(
                    block.feature_dim / block.mha.num_heads
                )
                for i in range(block.mha.num_heads):
                    W_q_inf = jnp.linalg.norm(
                        block.mha.query.kernel.value[:, i, :], jnp.inf
                    )
                    W_k_inf = jnp.linalg.norm(
                        block.mha.key.kernel.value[:, i, :], jnp.inf
                    )
                    W_v_inf = jnp.linalg.norm(
                        block.mha.value.kernel.value[:, i, :], jnp.inf
                    )
                    head_term += D * W_q_inf * W_k_inf * W_v_inf
                    sum_w_v += W_v_inf

                W_o = jnp.linalg.norm(
                    block.mha.out.kernel.value.reshape(
                        -1, block.mha.out.kernel.value.shape[-1]
                    ),
                    jnp.inf,
                )

                W1_inf = jnp.linalg.norm(block.ffn_dense1.kernel.value, jnp.inf)
                W2_inf = jnp.linalg.norm(block.ffn_dense2.kernel.value, jnp.inf)

                term1 = 0.5 + W1_inf * W2_inf
                term2 = (gamma1_inf / epsilon) * (0.5 + 0.5 * W_o * head_term)
                A_delta = term1 * term2

                L_MHA_u = W_o * (sum_w_v + 0.5 * head_term)
                B_delta = (gamma1_inf / epsilon) * (0.5 + W1_inf * W2_inf) * L_MHA_u

                # Cap to avoid extreme regularizer gradient explosion
                A_delta = jnp.minimum(A_delta, 3000.0)
                B_delta = jnp.minimum(B_delta, 3000.0)

                if solve_fn is not None:
                    # -------------------------------------------------------
                    # Full contraction-metric stability condition
                    # -------------------------------------------------------
                    # Encode observations through the actor to get raw MPC weights
                    # physical_state: (batch, num_agents, 2)
                    # reference: (batch, num_agents, 2)
                    batch_size = physical_state.shape[0]
                    num_agents = physical_state.shape[1]

                    # Re-run the actor encoder to get the raw weights matrix
                    local_obs = traj_batch.obs["local_obs"]
                    activation = (
                        nnx.relu
                        if actor_model.activation_name == "relu"
                        else nnx.tanh
                    )

                    if local_obs.ndim == 4:
                        x_enc = local_obs.reshape(
                            batch_size * num_agents,
                            actor_model.obs_seq_len,
                            actor_model.local_obs_dim,
                        )
                    elif local_obs.ndim == 3:
                        x_enc = local_obs.reshape(
                            batch_size * num_agents, actor_model.local_obs_dim
                        )
                    else:
                        x_enc = local_obs

                    encoded = actor_model._encode(x_enc, training=True)
                    if encoded.ndim == 3:
                        encoded = encoded[:, -1, :]

                    for hidden_layer in actor_model.actor_hidden:
                        encoded = activation(hidden_layer(encoded))
                    weights_matrix = actor_model.output_layer(encoded)
                    # (batch*num_agents, mpc_weight_dim)

                    # Flatten agents into batch for per-agent metric computation
                    weights_flat = weights_matrix  # (batch*num_agents, mpc_weight_dim)
                    phys_flat = physical_state.reshape(
                        batch_size * num_agents, nx
                    )  # (batch*num_agents, 2)
                    ref_flat = reference.reshape(
                        batch_size * num_agents, nx
                    )  # (batch*num_agents, 2)

                    def get_linearizations_and_metric(w, p_state, ref):
                        w_stop = jax.lax.stop_gradient(w)
                        p_state_stop = jax.lax.stop_gradient(p_state)
                        ref_stop = jax.lax.stop_gradient(ref)

                        _, X, U = solve_mpc(
                            w_stop,
                            p_state_stop,
                            ref_stop,
                            env_params,
                            env_dynamics,
                            solve_fn,
                            horizon,
                            dt,
                            nx,
                            nu,
                            mpc_iters=config.get("MPC_ITERS", 1),
                            return_trajectory=True,
                        )

                        # Linearize single-integrator dynamics: x_{k+1} = x_k + dt * v_max * u_k
                        # For midpoint integration: effectively A = I, B = v_max * dt * I
                        # But we compute it via autodiff for correctness
                        def single_step_dyn(x, u):
                            dx = env_dynamics.state_dot(x, u, None)
                            x_mid = x + (dt / 2.0) * dx
                            dx_mid = env_dynamics.state_dot(x_mid, u, None)
                            x_next = x + dt * dx_mid
                            return x_next

                        def step_error_dyn(delta_x, delta_u, x_nom, u_nom):
                            x_p = x_nom + delta_x
                            u_p = u_nom + delta_u
                            x_next_p = single_step_dyn(x_p, u_p)
                            x_next_nom = single_step_dyn(x_nom, u_nom)
                            return x_next_p - x_next_nom

                        def get_AB(x_nom, u_nom):
                            jac_fn = jax.jacfwd(step_error_dyn, argnums=(0, 1))
                            return jac_fn(
                                jnp.zeros(nx), jnp.zeros(nu), x_nom, u_nom
                            )

                        # A_seq: (horizon, nx, nx), B_seq: (horizon, nx, nu)
                        A_seq, B_seq = jax.vmap(get_AB)(X[:horizon], U)

                        # Decode Q, R from weights
                        half_dim = horizon * (nx + nu)
                        Q_R_traj = (
                            jnp.clip(
                                jax.nn.softplus(w[:half_dim]) * 10.0, 0.0, 100.0
                            )
                        )
                        Q_R_traj = Q_R_traj.reshape((horizon, nx + nu))
                        Q_seq = jax.vmap(jnp.diag)(Q_R_traj[:, :nx] + 0.1)
                        R_seq = jax.vmap(jnp.diag)(Q_R_traj[:, nx:] + 0.1)

                        # Compute contraction metric
                        M_0 = compute_contraction_metric_pure_jax(
                            A_seq, B_seq, Q_seq, R_seq
                        )

                        eigenvalues, _ = jnp.linalg.eigh(M_0)
                        eps = 1e-6
                        c_min = jnp.sqrt(jnp.maximum(eigenvalues[0], eps))
                        c_max = jnp.sqrt(jnp.maximum(eigenvalues[-1], eps))

                        return c_min, c_max, M_0, B_seq[0]

                    c_min, c_max, M_0, B_0 = jax.vmap(
                        get_linearizations_and_metric
                    )(weights_flat, phys_flat, ref_flat)

                    # Compute small-gain stability margin
                    rho_mpc = jnp.sqrt(
                        jnp.maximum(1 - 0.1 / (c_max**2 + 1e-6), 1e-6)
                    )

                    # Calculate S_Y_mix (coupling sensitivity through B)
                    B_2 = jnp.linalg.norm(B_0, 2, axis=(-1, -2))
                    S_Y_mix = B_2

                    # Calculate L_Gamma: Lipschitz constant of input projection
                    L_Gamma = jnp.linalg.norm(
                        actor_model.input_projection.kernel.value, jnp.inf
                    )

                    # Calculate L_Psi: Lipschitz constant of post-transformer actor head
                    L_Psi = 1.0
                    for hidden_layer in actor_model.actor_hidden:
                        L_Psi *= jnp.linalg.norm(
                            hidden_layer.kernel.value, jnp.inf
                        )
                    L_Psi *= jnp.linalg.norm(
                        actor_model.output_layer.kernel.value, jnp.inf
                    )

                    L_CL = rho_mpc
                    gamma_z = S_Y_mix * L_Psi * A_delta * c_max

                    small_gain_margin = (1.0 - L_CL) * (1.0 - A_delta) - (
                        gamma_z * B_delta * L_Gamma / (c_min + 1e-6)
                    )

                    small_gain_margin = jnp.where(
                        jnp.isnan(small_gain_margin), 3000.0, small_gain_margin
                    )
                    A_delta = jnp.where(jnp.isnan(A_delta), 3000.0, A_delta)

                    reg_loss += jnp.mean(
                        jnp.maximum(0.0, 0.01 - small_gain_margin)
                    )
                else:
                    reg_loss += jnp.maximum(0.0, A_delta - 0.8)

        reg_coef = config.get("REG_COEF", 0.01)
        reg_loss = jnp.minimum(reg_loss, 3e5)

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
            + reg_coef * reg_loss
        )
        return total_loss, (
            value_loss,
            loss_actor,
            entropy,
            old_approx_kl,
            approx_kl,
            clipfracs,
            reg_loss,
        )

    return _loss_fn


def make_update_fn(config, loss_fn, shuffle_batch_fn, networks):
    def _update_epoch(update_state: UpdateState, unused):
        def _update_minbatch(carry, batch_info):
            train_state, keep_training = carry
            traj_batch, advantages, targets = batch_info
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )

            approx_kl = total_loss[1][4]
            target_kl = config.get("TARGET_KL", 0.13)

            keep_training = jnp.logical_and(
                keep_training, approx_kl <= 1.5 * target_kl
            )

            new_train_state = jax.lax.cond(
                keep_training,
                lambda ts: ts.apply_gradients(grads=grads),
                lambda ts: ts,
                train_state,
            )
            return (new_train_state, keep_training), total_loss

        rng, _rng = jax.random.split(update_state.rng)
        minibatches = shuffle_batch_fn(
            _rng,
            (
                update_state.traj_batch,
                update_state.advantages,
                update_state.targets,
            ),
            config,
        )
        (train_state, _), total_loss = jax.lax.scan(
            _update_minbatch,
            (update_state.train_state, jnp.array(True)),
            minibatches,
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
    rngs_state = networks.get("rngs_state", {})
    critic_rngs = rngs_state.get("critic", {})
    critic_model = nnx.merge(
        networks["graphdef"]["critic"],
        collect_state.train_state.params["critic"],
        critic_rngs,
    )
    last_val = critic_model(collect_state.last_obs["global_state"])
    advantages, targets = gae_standard(config, traj_batch, last_val)
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


def extract_losses(loss_info):
    total_loss, aux = loss_info
    value_loss, loss_actor, entropy, old_approx_kl, approx_kl, clipfracs, reg_loss = (
        aux
    )

    losses = {
        "total": total_loss.mean(),
        "value": value_loss.mean(),
        "actor": loss_actor.mean(),
        "entropy": entropy.mean(),
        "old_approx_kl": old_approx_kl.mean(),
        "approx_kl": approx_kl.mean(),
        "clipfrac": clipfracs.mean(),
    }
    return losses, {"reg": reg_loss.mean()}


def init_networks(config, env, env_params, rng, load_params_fn=None):
    obs_space = env.observation_space(env_params)
    local_obs_shape = obs_space.spaces["local_obs"].shape
    if len(local_obs_shape) == 3:
        num_agents, obs_seq_len, local_obs_dim = local_obs_shape
    else:
        num_agents, local_obs_dim = local_obs_shape
        obs_seq_len = 1

    global_state_dim = obs_space.spaces["global_state"].shape[-1]

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
        return mpc_layer(weights, physical_state, reference), (
            weights,
            physical_state,
            reference,
        )

    def safe_mpc_layer_bwd(res, g):
        weights, physical_state, reference = res

        g = jnp.clip(g, -1.0, 1.0)
        g = jnp.where(jnp.abs(g) < 1e-10, 1e-10 * jnp.sign(g + 1e-20), g)

        _, vjp_fn = jax.vjp(mpc_layer, weights, physical_state, reference)
        g_w, g_p, g_r = vjp_fn(g)

        g_w = jax.tree_util.tree_map(
            lambda x: jnp.clip(
                jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0
            ),
            g_w,
        )
        g_p = jax.tree_util.tree_map(
            lambda x: (
                jnp.clip(
                    jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0
                )
                if x is not None
                else None
            ),
            g_p,
        )
        g_r = jax.tree_util.tree_map(
            lambda x: (
                jnp.clip(
                    jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0
                )
                if x is not None
                else None
            ),
            g_r,
        )

        return g_w, g_p, g_r

    safe_mpc_layer.defvjp(safe_mpc_layer_fwd, safe_mpc_layer_bwd)

    # Vmap over agents! (weights, pos, target are per-agent)
    vmap_mpc_layer_agents = jax.vmap(safe_mpc_layer, in_axes=(0, 0, 0))
    # Vmap over batch
    vmap_mpc_layer_batch = jax.vmap(vmap_mpc_layer_agents, in_axes=(0, 0, 0))

    actor_network = MAPPOTransformerDiffMPCActor(
        local_obs_dim=local_obs_dim,
        mpc_weight_dim=mpc_weight_dim,
        env_action_dim_per_agent=env_action_dim_per_agent,
        num_agents=num_agents,
        mpc_fn=vmap_mpc_layer_batch,
        obs_seq_len=obs_seq_len,
        activation=config["ACTIVATION"],
        transformer_hidden_dim=config.get("TRANSFORMER_HIDDEN_DIM", 128),
        num_heads=config.get("TRANSFORMER_NUM_HEADS", 2),
        num_encoder_layers=config.get("TRANSFORMER_NUM_LAYERS", 1),
        ffn_hidden_dim=config.get("TRANSFORMER_FFN_HIDDEN_DIM", 128),
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

    actor_params = actor_state.filter(nnx.Param)
    actor_rngs_state = actor_state.filter(
        lambda path, value: not isinstance(value, nnx.Param)
    )

    critic_params = critic_state.filter(nnx.Param)
    critic_rngs_state = critic_state.filter(
        lambda path, value: not isinstance(value, nnx.Param)
    )

    linear_schedule = make_linear_schedule(config)
    tx = create_optimizer(config, linear_schedule)

    train_state = TrainState.create(
        apply_fn=None,
        params={"actor": actor_params, "critic": critic_params},
        tx=tx,
    )

    graphdef = {"actor": actor_graphdef, "critic": critic_graphdef}

    if load_params_fn is not None:
        train_state = train_state.replace(params=load_params_fn(train_state.params))

    return {
        "graphdef": graphdef,
        "train_state": train_state,
        "rngs_state": {"actor": actor_rngs_state, "critic": critic_rngs_state},
        "state_template": {"actor": actor_state, "critic": critic_state},
    }


def make_eval_step(config, graphdef, env, env_params, rngs_state=None, **kwargs):
    """Create deterministic env-step function for DIFFMPC evaluation rollouts."""
    if rngs_state is None:
        rngs_state = {}
    actor_rngs = rngs_state.get("actor", {})

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
        actor_model = nnx.merge(
            graphdef["actor"],
            eval_state.train_state.params["actor"],
            actor_rngs,
        )

        physical_state, reference = get_physical_state(eval_state.env_state)

        pi = actor_model(
            eval_state.last_obs["local_obs"],
            physical_state=physical_state,
            reference=reference,
        )
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
    algo_name="MULTI_AGENT_DIFFMPC_TRANSFORMER_STAB",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    extract_losses=extract_losses,
    init_networks=init_networks,
    make_eval_step=make_eval_step,
)
