import jax
import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Unified rollout types for all PPO-family algorithms.
#
# Algorithm-specific fields live in the `extras` dict, which is a valid JAX
# pytree container.  Each algorithm populates a fixed set of keys so the
# structure stays constant within a single JIT-compiled training function.
#
# Examples:
#   PPO / RAPPO  : extras = {}
#   CPPO         : extras = {"nu": ..., "cvarlam": ...}
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """Rollout transition for all PPO-family algorithms."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    extras: dict = {}


class RunnerState(NamedTuple):
    """Runner state for all PPO-family algorithms."""
    train_state: object  # Flax TrainState
    env_state: object
    last_obs: jnp.ndarray
    rng: jnp.ndarray
    extras: dict = {}


class UpdateState(NamedTuple):
    """Update state for all PPO-family algorithms."""
    train_state: object
    traj_batch: object
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray
    extras: dict = {}


# ---------------------------------------------------------------------------
# Legacy aliases – kept so that scripts in legacy/ keep working.
# New algorithm code should use Transition / RunnerState / UpdateState.
# ---------------------------------------------------------------------------
RolloutTransition = Transition

class ReplayTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray

    def init_mean(self):
        return jnp.array(0.0)

    def init_var(self):
        return jnp.array(0.0)

    def compute_mean_std(self, current_mean, current_var, new_size):
        delta = current_mean[0] - self.reward
        new_mean = current_mean[0] + delta * self.reward.shape[0] / new_size
        m_a = current_var[0] * new_size
        m_b = jnp.var(self.reward) * self.reward.shape[0]
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * (new_size - self.reward.shape[0]) / new_size
        )
        new_var = M2 / new_size

        return jnp.array(new_mean), jnp.array(new_var)


class ReplayBufferState(NamedTuple):
    transitions: ReplayTransition  # shape [buffer_size, ...]
    position: jnp.ndarray  # scalar int32
    size: jnp.ndarray  # scalar int32
    mean: jnp.ndarray  # mean of the transitions
    var: jnp.ndarray  # standard deviation of the transitions


@jax.jit
def add_batch_to_buffer(
    buffer_state: ReplayBufferState,
    new_transitions: ReplayTransition,
) -> ReplayBufferState:
    batch_size = new_transitions.obs.shape[0]
    buffer_size = buffer_state.transitions.obs.shape[0]

    indices = (jnp.arange(batch_size) + buffer_state.position) % buffer_size

    # update each field in transitions
    updated_transitions = jax.tree_util.tree_map(
        lambda buf, new: buf.at[indices].set(new),
        buffer_state.transitions,
        new_transitions,
    )

    new_position = (buffer_state.position + batch_size) % buffer_size
    new_size = jnp.minimum(buffer_state.size + batch_size, buffer_size)
    # Update mean and std of the transitions
    new_mean, new_var = new_transitions.compute_mean_std(
        buffer_state.mean, buffer_state.var, new_size
    )

    return ReplayBufferState(
        updated_transitions, new_position, new_size, new_mean, new_var
    )


def sample_batch_from_buffer(
    buffer_state: ReplayBufferState, batch_size: int, rng: jax.Array
) -> ReplayTransition:
    max_index = buffer_state.size
    rng, key = jax.random.split(rng)
    indices = jax.random.randint(key, (batch_size,), 0, max_index)

    sampled_transitions = jax.tree_util.tree_map(
        lambda data: data[indices], buffer_state.transitions
    )
    return sampled_transitions


def create_empty_buffer(
    buffer_size: int,
    example_transition: ReplayTransition,
) -> ReplayBufferState:
    """Create an empty buffer state given a sample transition for shape inference"""
    empty_transitions = jax.tree_util.tree_map(
        lambda x: jnp.zeros((buffer_size,) + x.shape[1:], dtype=x.dtype),
        example_transition,
    )

    return ReplayBufferState(
        empty_transitions,
        jnp.array(0, jnp.int32),
        jnp.array(0, jnp.int32),
        example_transition.init_mean(),
        example_transition.init_var(),
    )
