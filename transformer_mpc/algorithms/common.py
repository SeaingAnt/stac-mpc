import jax
import jax.numpy as jnp

__all__ = ["gae_standard"]


def gae_standard(config, traj_batch, last_val):
    """Standard GAE for PPO-style returns.
    Returns (advantages, targets).
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        
        # Determine discount multiplier:
        # If the environment provides a specific 'discount' in info (e.g. to handle truncation), use it.
        # Otherwise fallback to (1 - done).
        if "discount" in transition.info:
            discount_mult = transition.info["discount"]
            # Broadcast to match done shape if needed (e.g. for RAPPO)
            while len(discount_mult.shape) < len(transition.done.shape):
                discount_mult = jnp.expand_dims(discount_mult, axis=-1)
        else:
            discount_mult = 1.0 - transition.done
        
        # When truncating, 'done' is true but 'discount' is > 0.
        # In this step, 'next_value' evaluates the post-reset initial state, not the terminal state.
        # So we bootstrap using 'real_next_value' from the un-reset observation.
        cond = (transition.done > 0.5) & (discount_mult > 0.5)
        value_to_bootstrap = jnp.where(
            cond,
            transition.info.get("real_next_value", next_value),
            next_value
        )
        
        delta = (
            transition.reward
            + config["GAMMA"] * value_to_bootstrap * discount_mult
            - transition.value
        )
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * discount_mult * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets
