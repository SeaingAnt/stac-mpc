import optax

__all__ = ["create_optimizer", "make_linear_schedule"]


def create_optimizer(config, linear_schedule_fn):
    """Create optimizer with optional learning rate annealing and weight decay."""
    
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=linear_schedule_fn, eps=1e-5, weight_decay=1e-4),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=config["LR"], eps=1e-5, weight_decay=1e-4),
        )
    return tx


def make_linear_schedule(config):
    """Create linear learning rate schedule."""
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    return linear_schedule
