import os
import sys

import jax
import jax.numpy as jnp
from types import SimpleNamespace

# Import from transformer_mpc
from transformer_mpc.env.drone_v0 import DroneV0, EnvParams, ModelState
from transformer_mpc.env.math import skew, quat2rotm, quat_product

import mpx.utils.mpc_wrapper as base_mpc_wrapper
from mpx.jax_ocp_solvers.jax_ocp_solvers import optimizers

def run_drone_mpc_test():
    env = DroneV0()
    params = env.default_params
    nx = env.state_dim
    nu = env.num_actions

    horizon = 20
    dt = 0.02
    
    # Base weights
    Q_base = jnp.zeros(13)
    Q_base = Q_base.at[0:2].set(50.0)   # px, py (Horizontal)
    Q_base = Q_base.at[2].set(50.0)     # pz (Aggressively prioritize altitude!)
    Q_base = Q_base.at[3:7].set(10.0)   # att (Keep it stable)
    Q_base = Q_base.at[7:10].set(2.0)   # vel
    Q_base = Q_base.at[10:13].set(1.0)  # omega

    Q_traj = jnp.tile(Q_base, (horizon + 1, 1))
    decay = jnp.linspace(1.0, 0.1, horizon + 1)[:, None]
    Q_traj = Q_traj.at[:, 0:3].multiply(decay)

    # Create time-varying R:
    R_base = jnp.ones(4) * 0.01 
    R_traj = jnp.tile(R_base, (horizon + 1, 1))
    increase = jnp.linspace(1.0, 10.0, horizon + 1)[:, None]
    R_traj = R_traj * increase

    # Create linear term P:
    P_traj = jnp.zeros((horizon + 1, 13 + 4))
    #P_traj = P_traj.at[:, 0].set(-5.0) 

    # Final State cost weights (quadratic)
    Q_final = Q_base * 10.0
    P_final = jnp.zeros(13)
    #P_final = P_final.at[0].set(-50.0) 

    # Initial state (hover slightly off target)
    initial_state = jnp.array([
        0.5, 0.5, 0.0,           
        1.0, 0.0, 0.0, 0.0,      
        0.0, 0.0, 0.0,           
        0.0, 0.0, 0.0            
    ])
    
    # Target state
    target_state = jnp.array([
        0.0, 0.0, 0.5,           
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])

    nominal_hover = jnp.array([params.m * params.g, 0.0, 0.0, 0.0])


    def state_dot(state: jax.Array, control: jax.Array) -> jax.Array:
        # State: 0:3 = pos, 3:7 = quat, 7:10 = vel, 10:13 = omega
        pos = state[0:3]
        quat = state[3:7]
        vel = state[7:10]
        omega = state[10:13]

        # Normalize quaternion to avoid blowup
        quat = quat / (jnp.linalg.norm(quat) + 1e-6)

        u = jnp.clip(control, params.min_input, params.max_input)
        orientation_mat = quat2rotm(quat)
        ang_vel_skew = skew(omega)

        total_force = (
            jnp.array([0.0, 0.0, u[0]])
            - params.m * params.g * orientation_mat[2, :]
            - params.m * ang_vel_skew @ ang_vel_skew @ params.com_pos
        )

        total_torque = (
            u[1:4]
            - ang_vel_skew @ params.I @ omega
            - params.m * params.g * skew(params.com_pos) @ orientation_mat[2, :]
        )

        si_mat = jnp.diag(
            jnp.concatenate([params.m * jnp.ones(3, dtype=jnp.float32), params.I])
        )
        si_mat = si_mat.at[0:3, 3:6].set(-params.m * skew(params.com_pos))
        si_mat = si_mat.at[3:6, 0:3].set(params.m * skew(params.com_pos))

        acc = jnp.linalg.solve(si_mat, jnp.concatenate([total_force, total_torque]))
        
        acc_lin = orientation_mat @ acc[:3]
        acc_ang = acc[3:6]

        pos_dot = vel
        omega_quat = jnp.array([0.0, omega[0], omega[1], omega[2]])
        quat_dot = 0.5 * quat_product(quat, omega_quat)

        return jnp.concatenate([pos_dot, quat_dot, acc_lin, acc_ang])


    def dynamics(x, u, t, parameter):
        # Discretization scheme = 2 is midpoint in DiffMPC
        dx = state_dot(x, u)
        x_mid = x + (dt / 2.0) * dx
        x_mid = x_mid.at[3:7].set(x_mid[3:7] / (jnp.linalg.norm(x_mid[3:7]) + 1e-6))
        dx_mid = state_dot(x_mid, u)
        next_x = x + dt * dx_mid
        return next_x


    def cost(W, reference, x, u, t):
        # DiffMPC penalization formulation
        x_diff = x - target_state
        u_diff = u - nominal_hover

        Qt = Q_traj[t]
        Rt = R_traj[t]
        Pt = P_traj[t]

        P_x = Pt[0:13]
        P_u = Pt[13:]

        stage_cost = 0.5 * jnp.sum(Qt * (x_diff ** 2)) + jnp.sum(P_x * x) + \
                     0.5 * jnp.sum(Rt * (u_diff ** 2)) + jnp.sum(P_u * u)

        term_cost = 0.5 * jnp.sum(Q_final * (x_diff ** 2)) + jnp.sum(P_final * x)

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
    
    W = jnp.zeros((horizon, 1))
    reference = jnp.zeros((horizon + 1,))
    parameter = jnp.zeros((horizon + 1,))

    current_state = initial_state
    sim_steps = 200

    print("--- Starting MPC Closed-Loop ---")
    for step in range(sim_steps):
        
        sol_X, sol_U, sol_V = solve_fn(reference, parameter, W, current_state, init_X0, init_U0, init_V0)
        
        current_state = sol_X[1]
        
        # Warm start next iteration by shifting the trajectory
        shifted_X = jnp.concatenate([sol_X[1:], sol_X[-1:]], axis=0)
        shifted_U = jnp.concatenate([sol_U[1:], sol_U[-1:]], axis=0)
        init_X0 = shifted_X
        init_U0 = shifted_U
        init_V0 = sol_V

    print("--- Simulation Complete ---")
    final_pos_str = f"[{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}]"
    print(f"Final Pos: {final_pos_str}")

if __name__ == "__main__":
    run_drone_mpc_test()