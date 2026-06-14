import os
import sys

import jax
import jax.numpy as jnp
from types import SimpleNamespace

# Import from transformer_mpc
from transformer_mpc.env.drone_v2 import DroneV2
from transformer_mpc.env.math import skew, quat2rotm, quat_product

import mpx.utils.mpc_wrapper as base_mpc_wrapper


def _safe_control_violation(u, min_input, max_input):
    lower = jax.nn.softplus(10.0 * (min_input - u)) / 10.0
    upper = jax.nn.softplus(10.0 * (u - max_input)) / 10.0
    return lower + upper


def run_drone_v2_mpc_test():
    env = DroneV2()
    params = env.default_params
    nx = env.state_dim
    nu = env.num_actions

    horizon = 20
    dt = params.dt
    
    # Base weights similar to diffmpc_transformer_stab interpretation
    Q_base = jnp.zeros(12)  # nx_err = 12 (3 pos + 3 rot + 3 vel + 3 omega)
    Q_base = Q_base.at[0:2].set(80.0)   # px, py
    Q_base = Q_base.at[2].set(80.0)     # pz
    Q_base = Q_base.at[3:6].set(10.0)   # att error
    Q_base = Q_base.at[6:9].set(2.0)    # vel
    Q_base = Q_base.at[9:12].set(1.0)   # omega

    Q_traj = jnp.tile(Q_base, (horizon + 1, 1))
    decay = jnp.linspace(1.0, 1.0, horizon + 1)[:, None]
    Q_traj = Q_traj.at[:, 0:3].multiply(decay)

    # R and P
    R_base = jnp.ones(4) * 0.01 
    R_traj = jnp.tile(R_base, (horizon + 1, 1))
    increase = jnp.linspace(1.0, 10.0, horizon + 1)[:, None]
    R_traj = R_traj * increase

    P_traj = jnp.zeros((horizon + 1, 12 + 4))
    
    # We want u_steady = nominal_hover. 
    # To have minimum at u = nominal_hover with cost 0.5 * R * u^2 + P_u * u:
    # P_u = - R * nominal_hover
    nominal_hover = jnp.array([params.m * params.g, 0.0, 0.0, 0.0])
    for i in range(horizon + 1):
        P_traj = P_traj.at[i, 12:].set(-R_traj[i] * nominal_hover)

    # Initial state
    initial_state = jnp.array([
        0.0, -1.0, 0.5,           
        1.0, 0.0, 0.0, 0.0,      
        0.0, 0.0, 0.0,           
        0.0, 0.0, 0.0            
    ])
    
    # Target state
    target_state = jnp.array([
        0.0, 1.0, 0.5,           
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    obstacle_pos = jnp.array([0.0, 0.0, 0.5])
    
    env_dynamics = env.mpc_dynamics(nx, nu, params)

    def dynamics(x, u, t, parameter):
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
        return x_next

    def cost(W, reference, x, u, t):
        base_env_state = reference
        nx_err = nx - 1

        # We compute error with respect to target_state
        # pos error
        pos_err = x[:3] - target_state[:3]
        quat = x[3:7]
        x_following_err = x[7:] - target_state[7:]

        # For attitude error, we use the same math as diffmpc_transformer_stab
        # But we compare against target_state's quaternion (which is [1, 0, 0, 0])
        target_quat = target_state[3:7] 
        target_quat_inv = jnp.array([target_quat[0], -target_quat[1], -target_quat[2], -target_quat[3]])
        # In diffmpc_transformer_stab, they do: q_err = quat_product(target_quat_inv, quat)
        q_err = quat_product(target_quat_inv, quat)
        
        sign_w = jnp.sign(q_err[0])
        sign_w = jnp.where(sign_w == 0, 1.0, sign_w) 
        
        error_rot = 2.0 * sign_w * q_err[1:4]
        
        x_w_quat_error = jnp.concatenate([pos_err, error_rot, x_following_err], axis=-1)
        
        idx = jnp.minimum(t, horizon - 1)
        Qt = Q_traj[idx]
        Rt = R_traj[idx]
        Pt = P_traj[idx]

        P_x = Pt[:nx_err]
        P_u = Pt[nx_err:]

        control_violation = _safe_control_violation(
            u,
            params.min_input,
            params.max_input,
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
            # The constraint cost in drone_v2 uses x and base_env_state[:3] as obstacle_pos
            c_cost = env_dynamics.constraint_cost(x, u, t, base_env_state, params)
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
        limited_memory=False
    )
    
    solve_fn = jax.jit(solve_fn)

    init_X0 = jnp.tile(initial_state, (horizon + 1, 1))
    init_U0 = jnp.tile(nominal_hover, (horizon, 1))
    init_V0 = jnp.zeros((horizon + 1, nx))
    
    W = jnp.zeros(1) # dummy W
    reference = jnp.concatenate([obstacle_pos, jnp.zeros(10)]) # base_env_state should have enough length if needed
    parameter = jnp.zeros(1)

    current_state = initial_state
    sim_steps = 200

    print("--- Starting MPC Closed-Loop ---")
    
    # Store trajectory for optional plotting
    traj = []

    for step in range(sim_steps):
        traj.append(current_state[:3])
        sol_X, sol_U, sol_V = solve_fn(reference, parameter, W, current_state, init_X0, init_U0, init_V0)
        
        current_state = sol_X[1]
        
        shifted_X = jnp.concatenate([sol_X[1:], sol_X[-1:]], axis=0)
        shifted_U = jnp.concatenate([sol_U[1:], sol_U[-1:]], axis=0)
        init_X0 = shifted_X
        init_U0 = shifted_U
        init_V0 = sol_V

    traj.append(current_state[:3])
    print("--- Simulation Complete ---")
    final_pos_str = f"[{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}]"
    print(f"Final Pos: {final_pos_str}")
    
    # Check if constraint was roughly respected (dist to obstacle > obstacle_radius)
    min_dist = min([jnp.linalg.norm(p - obstacle_pos) for p in traj])
    print(f"Minimum distance to obstacle: {min_dist:.3f} (Radius: {params.obstacle_radius})")

if __name__ == "__main__":
    run_drone_v2_mpc_test()