""" "DiffMPC Algorithm implementation with Transformer architecture."""

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
from networks import TransformerActorCritic, ActorCritic
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
        norm_mid = jax.lax.stop_gradient(jnp.maximum(jnp.linalg.norm(x_mid[3:7]), 1e-6))
        x_mid = x_mid.at[3:7].set(x_mid[3:7] / norm_mid)
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        # Crucial for numerical stability: ensure the final predicted quaternion remains a unit quaternion
        # Otherwise, internal MPC states will drift over the MPC horizon and the dynamics will explode
        norm_next = jax.lax.stop_gradient(
            jnp.maximum(jnp.linalg.norm(x_next[3:7]), 1e-6)
        )
        x_next = x_next.at[3:7].set(x_next[3:7] / norm_next)
        return x_next

    def cost(W, reference, x, u, t):
        base_env_state = reference
        weights = W
        nx_err = nx - 1
        half_dim = horizon * (nx_err + nu)

        pos = x[:3]
        quat = x[3:7]
        x_following = x[7:]

        identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0]) 
        q_err = quat_product(identity_quat, quat)
        
        sign_w = jnp.sign(q_err[0])
        sign_w = jnp.where(sign_w == 0, 1.0, sign_w) 
        
        error_rot = 2.0 * sign_w * q_err[1:4]
        
        x_w_quat_error = jnp.concatenate([pos, error_rot, x_following], axis=-1)
        
        Q_R_traj = jax.nn.sigmoid(weights[:half_dim]) * 100.0
        p_logits = weights[half_dim:]

        Q_R_traj = Q_R_traj.reshape((horizon, nx_err + nu))
        # Keep the +0.05 minimum floor for strict positive definiteness
        Q_traj = Q_R_traj[:, :nx_err] + 0.25
        R_traj = Q_R_traj[:, nx_err:] + 0.01

        # P can naturally be negative, so we map it symmetrically
        p_mapped = 50.0 * jnp.tanh(p_logits)
        P_traj = p_mapped.reshape((horizon, nx_err + nu))

        idx = jnp.minimum(t, horizon - 1)
        Qt = Q_traj[idx]
        Rt = R_traj[idx]
        Pt = P_traj[idx]

        P_x = Pt[:nx_err]
        P_u = Pt[nx_err:]

        control_violation = _safe_control_violation(
            u,
            env_params.min_input,
            env_params.max_input,
        )
        control_constraint_cost = 25.0 * jnp.sum(control_violation**2)

        stage_cost = (
            0.5 * (jnp.sum(Qt * x_w_quat_error**2) + jnp.sum(Rt * u**2))
            + jnp.sum(P_x * x_w_quat_error)
            + jnp.sum(P_u * u)
            + control_constraint_cost
        )
        term_cost = 0.5 * jnp.sum(Qt * x_w_quat_error**2) + jnp.sum(P_x * x_w_quat_error)

        if hasattr(env_dynamics, 'constraint_cost'):
            c_cost = env_dynamics.constraint_cost(x, u, t, base_env_state, env_params)
            stage_cost += c_cost
            term_cost += c_cost

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
    env_params,
    solve_fn,
    env_dynamics,
    horizon,
    dt,
    nx,
    nu,
    mpc_iters=1,
    return_trajectory=False,
    reference=None,
):
    nominal_hover = jnp.array([env_params.m * env_params.g, 0.0, 0.0, 0.0])

    W = weights
    reference = reference if reference is not None else jnp.zeros(1)
    parameter = jnp.zeros(1)

    init_U0 = jnp.tile(nominal_hover, (horizon, 1))

    def rollout_step(x, u):
        dx = env_dynamics.state_dot(x, u, None)
        x_mid = x + (dt / 2.0) * dx
        norm_mid = jax.lax.stop_gradient(jnp.maximum(jnp.linalg.norm(x_mid[3:7]), 1e-6))
        x_mid = x_mid.at[3:7].set(x_mid[3:7] / norm_mid)
        dx_mid = env_dynamics.state_dot(x_mid, u, None)
        x_next = x + dt * dx_mid
        norm_next = jax.lax.stop_gradient(
            jnp.maximum(jnp.linalg.norm(x_next[3:7]), 1e-6)
        )
        x_next = x_next.at[3:7].set(x_next[3:7] / norm_next)
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
        (jnp.isnan(physical_action) | jnp.isinf(physical_action)).any(),
        nominal_hover,
        physical_action,
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

    if return_trajectory:
        return normalized_action, sol_X, sol_U
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
    rngs_state = networks.get("rngs_state", {})

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
        phys = jnp.concatenate([base.pos, base.attitude, base.vel, base.omega], axis=-1)
        if hasattr(base, 'obstacle_pos'):
            phys = jnp.concatenate([phys, base.obstacle_pos], axis=-1)
        return phys

    def _env_step(runner_state: RunnerState, unused):
        rng, _rng = jax.random.split(runner_state.rng)
        # Reconstruct full state using nnx.merge
        model = nnx.merge(graphdef, runner_state.train_state.params, rngs_state)

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


def compute_contraction_metric_pure_jax(
    A_seq, B_seq, Q_seq, R_seq, tol=1e-5, max_iters=50
):
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


def make_loss_fn(config, graphdef, rngs_state=None, state_template=None, **kwargs):
    if rngs_state is None:
        rngs_state = {}

    env = kwargs.get("env")
    env_params = kwargs.get("env_params")

    nx = env.state_dim if hasattr(env, "state_dim") else 13
    nu = env.num_actions if hasattr(env, "num_actions") else 4
    horizon = (
        getattr(env_params, "mpc_horizon", config.get("MPC_HORIZON", 10))
        if env_params
        else 10
    )
    dt = getattr(env_params, "dt", 0.02) if env_params else 0.02
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
        # Reconstruct full state by merging params and rngs_state
        model = nnx.merge(graphdef, params, rngs_state)

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

        # Prevent extreme PPO policy loss spikes / gradient explosions
        # when continuous action probability predictions jump significantly
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

        def soft_max(x, alpha=5000.0, axis=None):
            """
            Computes a numerically stable soft maximum.
            Higher alpha means higher accuracy to the true max, but steeper gradients.
            """
            return jax.nn.logsumexp(alpha * x, axis=axis) / alpha

        def soft_matrix_inf_norm(W, alpha=5000.0):
            """
            Soft induced infinity norm for a matrix.
            The true inf norm is the maximum absolute row sum: max_i(sum_j(|W_ij|))
            """
            row_sums = jnp.sum(jnp.abs(W), axis=-1)
            return soft_max(row_sums, alpha=alpha, axis=0)

        reg_loss = 0.0
        if hasattr(model, "encoder_blocks"):
            epsilon = 0.1
            for block in model.encoder_blocks:
                gamma1_inf = soft_max(jnp.abs(block.mha_ln.scale.value))
                gamma2_inf = soft_max(jnp.abs(block.ffn_ln.scale.value))

                head_term = 0.0
                sum_w_v = 0.0
                D = model.actor_seq_len/jnp.sqrt(block.feature_dim / block.mha.num_heads)
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

                # Cap A_delta and B_delta to avoid extreme regularizer gradient explosion
                A_delta = jnp.minimum(A_delta, 3000.0)
                B_delta = jnp.minimum(B_delta, 3000.0)

                if solve_fn is not None:
                    # Extract raw weights matrix natively to preserve gradients
                    encoded = model._encode(traj_batch.obs, training=True)
                    if encoded.ndim == 3:
                        encoded = encoded[:, -model.actor_seq_len :, :]
                        encoded = encoded.reshape(encoded.shape[0], -1)
                    activation = (
                        nnx.relu if model.activation_name == "relu" else nnx.tanh
                    )
                    for hidden_layer in model.actor_hidden:
                        encoded = activation(hidden_layer(encoded))
                    weights_matrix = model.actor_output(encoded)

                    def get_linearizations_and_metric(w, p_state):
                        w_stop = jax.lax.stop_gradient(w)
                        p_state_stop = jax.lax.stop_gradient(p_state)
                        real_p_state = p_state_stop[:nx]
                        reference = p_state_stop[nx:] if p_state_stop.shape[-1] > nx else None
                        _, X, U = solve_mpc(
                            w_stop,
                            real_p_state,
                            env_params,
                            solve_fn,
                            env_dynamics,
                            horizon,
                            dt,
                            nx,
                            nu,
                            mpc_iters=config.get("MPC_ITERS", 1),
                            return_trajectory=True,
                            reference=reference,
                        )

                        def single_step_dyn(x, u):
                            dx = env_dynamics.state_dot(x, u, None)
                            x_mid = x + (dt / 2.0) * dx
                            norm_mid = jax.lax.stop_gradient(jnp.maximum(jnp.linalg.norm(x_mid[3:7]), 1e-6))
                            x_mid = x_mid.at[3:7].set(x_mid[3:7] / norm_mid)
                            dx_mid = env_dynamics.state_dot(x_mid, u, None)
                            x_next = x + dt * dx_mid
                            norm_next = jax.lax.stop_gradient(jnp.maximum(jnp.linalg.norm(x_next[3:7]), 1e-6))
                            x_next = x_next.at[3:7].set(x_next[3:7] / norm_next)
                            return x_next

                        def apply_tangent_perturbation(x, delta_x):
                            pos = x[:3] + delta_x[:3]
                            quat = x[3:7]
                            delta_theta = delta_x[3:6]
                            
                            # Map small-angle 3D vector to quaternion perturbation [w, x, y, z]
                            dq_v = delta_theta / 2.0
                            # Evaluated at delta=0, sqrt(1-x) gradient is safely -0.5 (no NaN here)
                            dq_w = jnp.sqrt(jnp.maximum(1.0 - jnp.sum(dq_v**2), 1e-12)) 
                            dq = jnp.concatenate([jnp.array([dq_w]), dq_v])
                            
                            # Local frame rotation: current_quat * perturbation
                            q_new = quat_product(quat, dq)
                            q_new = q_new / jnp.linalg.norm(q_new)
                            
                            following = x[7:] + delta_x[6:]
                            return jnp.concatenate([pos, q_new, following])

                        def compute_tangent_error(x, x_ref):
                            pos_err = x[:3] - x_ref[:3]
                            
                            # q_err = x_ref^{-1} * x  (Assuming [w, x, y, z] convention)
                            q_ref_inv = jnp.array([x_ref[3], -x_ref[4], -x_ref[5], -x_ref[6]])
                            q_err = quat_product(q_ref_inv, x[3:7])
                            
                            # ML-Safe Hemisphere enforcement
                            sign_w = jnp.sign(q_err[0])
                            sign_w = jnp.where(sign_w == 0, 1.0, sign_w)
                            theta_err = 2.0 * sign_w * q_err[1:4]
                            
                            following_err = x[7:] - x_ref[7:]
                            return jnp.concatenate([pos_err, theta_err, following_err])

                        def step_error_dyn(delta_x, delta_u, x_nom, u_nom):
                            # Perturb the inputs in the tangent space
                            x_p = apply_tangent_perturbation(x_nom, delta_x)
                            u_p = u_nom + delta_u
                            
                            # Step both perturbed and nominal states through raw physics
                            x_next_p = single_step_dyn(x_p, u_p)
                            x_next_nom = single_step_dyn(x_nom, u_nom)
                            
                            # Compute the error between them in the tangent space
                            return compute_tangent_error(x_next_p, x_next_nom)

                        def get_AB(x_nom, u_nom):
                            nx_err = nx - 1
                            jac_fn = jax.jacfwd(step_error_dyn, argnums=(0, 1))
                            return jac_fn(jnp.zeros(nx_err), jnp.zeros(nu), x_nom, u_nom)

                        # A_seq is now (horizon, 12, 12), B_seq is (horizon, 12, 4)
                        A_seq, B_seq = jax.vmap(get_AB)(X[:horizon], U)

                        # 6. Decode weights using the EXACT nx_err dimension (12)
                        nx_err = nx - 1
                        half_dim = horizon * (nx_err + nu)
                        Q_R_traj = jax.nn.sigmoid(w[:half_dim]) * 100.0
                        Q_R_traj = Q_R_traj.reshape((horizon, nx_err + nu))
                        Q_seq = jax.vmap(jnp.diag)(Q_R_traj[:, :nx_err] + 0.25)
                        R_seq = jax.vmap(jnp.diag)(Q_R_traj[:, nx_err:] + 0.01)

                        # 7. Compute Riemannian Contraction Metric seamlessly
                        M_0 = compute_contraction_metric_pure_jax(
                            A_seq, B_seq, Q_seq, R_seq
                        )

                        eigenvalues, _ = jnp.linalg.eigh(M_0)
                        eps = 1e-6
                        c_min = jnp.sqrt(jnp.maximum(eigenvalues[0], eps))
                        c_max = jnp.sqrt(jnp.maximum(eigenvalues[-1], eps))
                        
                        return c_min, c_max, M_0, B_seq[0]

                    def norm_M(x, M):
                        eigenvalues, _ = jnp.linalg.eigh(x.T @ M @ x)
                        return jnp.sqrt(eigenvalues[-1])

                    c_min, c_max, M_0, B_0 = jax.vmap(get_linearizations_and_metric)(
                        weights_matrix, physical_state
                    )
                    # c_x = c_x_batch.mean()

                    # Compute small-gain stability margin
                    rho_mpc = jnp.sqrt(
                        jnp.maximum(1 - 0.25 / (c_max**2 + 1e-6), 1e-6)
                    )  # this one is sqrt(1 - q_min/ max_bound(M))

                    # Calculate S_Y_mix
                    B_2 = jnp.linalg.norm(B_0, 2, axis=(-1, -2))
                    S_Y_mix = B_2 / 0.5 * 53.0  # scale up to have a meaningful regularizer gradient signal, since B_2 can be very small

                    # Calculate L_Gamma: Lipschitz constant of pre-transformer input projection
                    L_Gamma = jnp.linalg.norm(
                        model.input_projection.kernel.value, jnp.inf
                    )

                    # Calculate L_Psi: Lipschitz constant of post-transformer actor head
                    L_Psi = 1.0
                    for hidden_layer in model.actor_hidden:
                        L_Psi *= jnp.linalg.norm(hidden_layer.kernel.value, jnp.inf)
                    L_Psi *= jnp.linalg.norm(model.actor_output.kernel.value, jnp.inf)

                    L_CL = rho_mpc
                    gamma_z = S_Y_mix * L_Psi * A_delta * c_max

                    small_gain_margin = (1.0 - L_CL) * (1.0 - A_delta) - (
                        gamma_z * B_delta * L_Gamma / (c_min + 1e-6)
                    )

                    small_gain_margin = jnp.where(jnp.isnan(small_gain_margin), 3000.0, small_gain_margin)
                    A_delta = jnp.where(jnp.isnan(A_delta), 3000.0, A_delta)

                    reg_loss += jnp.mean(jnp.maximum(0.0,0.1-small_gain_margin))
                    #reg_loss += jnp.mean(jnp.maximum(0.0, L_CL - 0.8))
                    #reg_loss += jnp.maximum(0.0, A_delta - 0.8)
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

            # Persist keep_training state so we stop the whole epoch once breached
            keep_training = jnp.logical_and(keep_training, approx_kl <= 1.5 * target_kl)

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
            (update_state.traj_batch, update_state.advantages, update_state.targets),
            config,
        )
        (train_state, _), total_loss = jax.lax.scan(
            _update_minbatch, (update_state.train_state, jnp.array(True)), minibatches
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
    graphdef = networks["graphdef"]
    rngs_state = networks.get("rngs_state", {})

    # Reconstruct model using nnx.merge
    model = nnx.merge(graphdef, collect_state.train_state.params, rngs_state)
    last_val = model.critic(collect_state.last_obs)
    advantages, targets = gae_standard(config, traj_batch, last_val)
    return {
        "advantages": advantages,
        "targets": targets,
    }, collect_state


def extract_losses(loss_info):
    total_loss, aux = loss_info
    value_loss, loss_actor, entropy, old_approx_kl, approx_kl, clipfracs, reg_loss = aux

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
    if len(obs_space.shape) != 2:
        raise ValueError(
            "Transformer DiffMPC expects a 2D observation space shaped as"
            " (sequence_length, feature_dim)."
        )
    _seq_len, obs_dim = obs_space.shape

    nx = env.state_dim if hasattr(env, "state_dim") else 13
    nu = env.num_actions if hasattr(env, "num_actions") else 4
    # Prefer MPC horizon from environment parameters; fall back to config
    horizon = getattr(env_params, "mpc_horizon", config.get("MPC_HORIZON", 10))

    # Each horizon step predicts one Q/R block and one P block.
    mpc_weight_dim = horizon * 2 * (nx + nu -1) # -1 because Lie algebra quaternion error is 3D, not 4D
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
        real_p_state = physical_state[:nx]
        reference = physical_state[nx:] if physical_state.shape[-1] > nx else None
        return solve_mpc(
            weights,
            real_p_state,
            env_params,
            solve_fn,
            env_dynamics,
            horizon,
            getattr(env_params, "dt", 0.02),
            nx,
            nu,
            mpc_iters=config.get("MPC_ITERS", 1),
            reference=reference,
        )

    @jax.custom_vjp
    def safe_mpc_layer(weights, physical_state):
        return mpc_layer(weights, physical_state)

    def safe_mpc_layer_fwd(weights, physical_state):
        return mpc_layer(weights, physical_state), (weights, physical_state)

    def safe_mpc_layer_bwd(res, g):
        weights, physical_state = res

        # Keep the MPC adjoint bounded so gradients remain numerically stable.
        g = jnp.clip(g, -1.0, 1.0)
        g = jnp.where(jnp.abs(g) < 1e-10, 1e-10 * jnp.sign(g + 1e-20), g)

        _, vjp_fn = jax.vjp(mpc_layer, weights, physical_state)
        g_w, g_p = vjp_fn(g)

        g_w = jax.tree_util.tree_map(
            lambda x: jnp.clip(
                jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0
            ),
            g_w,
        )
        g_p = jax.tree_util.tree_map(
            lambda x: (
                jnp.clip(jnp.where(jnp.isnan(x) | jnp.isinf(x), 0.0, x), -1.0, 1.0)
                if x is not None
                else None
            ),
            g_p,
        )
        return g_w, g_p

    safe_mpc_layer.defvjp(safe_mpc_layer_fwd, safe_mpc_layer_bwd)

    def vmap_wrapper(weights, physical_states):
        """Wrapper to handle both 2D (batch, flat) and 3D (batch, horizon, features) weights."""
        # If weights are 3D, reshape to 2D for vmap
        if weights.ndim == 3:
            batch_size, horizon_inner, features_per_token = weights.shape
            weights_2d = weights.reshape(batch_size, -1)
        else:
            weights_2d = weights

        # Apply vmap and reshape output
        return jax.vmap(safe_mpc_layer, in_axes=(0, 0))(weights_2d, physical_states)

    vmap_mpc_layer = vmap_wrapper

    network = TransformerActorCritic(
        obs_dim=obs_dim,
        action_dim=mpc_weight_dim,
        obs_seq_len=_seq_len,
        env_action_dim=env_action_dim,
        mpc_fn=vmap_mpc_layer,
        activation=config["ACTIVATION"],
        actor_head_hidden_dim=config["ACTOR_LAYER_SIZES"],
        critic_layer_sizes=config["CRITIC_LAYER_SIZES"],
        actor_seq_len=horizon,
        rngs=nnx.Rngs(rng),
    )

    # Split into graphdef and all state
    graphdef, state_all = nnx.split(network)
    # Extract params (trainable) and rngs (non-trainable) separately
    params = state_all.filter(nnx.Param)
    # Keep everything that's not a Param (RNGs, buffers, etc.)
    rngs_state = state_all.filter(lambda path, value: not isinstance(value, nnx.Param))

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
        "rngs_state": rngs_state,  # Store RNG state separately
        "state_template": state_all,
    }


SPEC = AlgorithmSpec(
    algo_name="DIFFMPC_TRANSFORMER_STAB",
    wrap_env=wrap_env,
    make_loss_fn=make_loss_fn,
    make_collect_fn=make_collect_fn,
    make_update_fn=make_update_fn,
    calculate_gae=calculate_gae,
    extract_losses=extract_losses,
    init_networks=init_networks,
)
