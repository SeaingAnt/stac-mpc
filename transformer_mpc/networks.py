from flax import nnx
from flax.linen.initializers import constant, orthogonal
import distrax
import jax
import jax.numpy as jnp


class ActorCritic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        env_action_dim: int = None,
        mpc_fn=None,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        critic_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mpc_fn = mpc_fn
        # If using MPC, the final log_std needs to match the physical control dim (e.g. 4), not the cost map dim
        self.env_action_dim = (
            env_action_dim if env_action_dim is not None else action_dim
        )
        self.activation_name = activation

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(self.env_action_dim))

        # Build critic layers
        _critic_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x, physical_state=None):
        """Full forward pass with actor and critic."""
        pi = self.actor(x, physical_state)
        value = self.critic(x)
        return pi, value

    def actor(self, x, physical_state=None):
        """Actor forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = x[:, -1, :] if x.ndim == 3 else x  # Use last obs for actor
        for layer in self.actor_layers:
            x = activation(layer(x))

        # Output is either the direct physical action or the 34D Cost Map weights
        out = self.actor_output(x)

        if self.mpc_fn is not None:
            # Pass the generated Cost Map weights and physical state into the differentiable solver
            actor_mean = self.mpc_fn(out, physical_state)
        else:
            actor_mean = out

        # Clip log_std to avoid numerical instability
        log_std = jnp.clip(self.actor_log_std.value, -5.0, 2.0)

        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))

    def critic(self, x):
        """Critic forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = x[:, -1, :] if x.ndim == 3 else x  # Use raw obs for critic
        for layer in self.critic_layers:
            x = activation(layer(x))
        return jnp.squeeze(self.critic_output_layer(x), axis=-1)


class ActorDCritic(nnx.Module):
    """Actor-Critic with distributional critic using cosine embeddings for quantile regression."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        env_action_dim: int = None,
        mpc_fn=None,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        hidden_size: int = 32,
        n_cosines: int = 64,
        n_taus: int = 64,
        lb: float = 0.0,
        ub: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mpc_fn = mpc_fn
        self.env_action_dim = (
            env_action_dim if env_action_dim is not None else action_dim
        )
        self.activation_name = activation
        self.hidden_size = hidden_size
        self.n_cosines = n_cosines
        self.n_taus = n_taus
        self.lb = lb
        self.ub = ub

        # Build actor layers
        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(self.env_action_dim))

        # Build critic layers
        self.cosine_embedding = nnx.Linear(
            n_cosines,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.state_embedding = nnx.Linear(
            obs_dim,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            hidden_size,
            hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.fc3 = nnx.Linear(
            hidden_size,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, state, rng_key=None, physical_state=None):
        """Full forward pass with actor and distributional critic."""
        pi = self.actor(state, physical_state)
        quantile_values, taus = self.critic(state, rng_key)
        return pi, (quantile_values, taus)

    def actor(self, x, physical_state=None):
        """Actor forward pass."""
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.actor_layers:
            x = activation(layer(x))
        out = self.actor_output(x)

        if self.mpc_fn is not None:
            actor_mean = self.mpc_fn(out, physical_state)
        else:
            actor_mean = out

        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(jnp.clip(self.actor_log_std.value, -5.0, 2.0))
        )

    def critic(self, state, rng_key=None):
        """Distributional critic forward pass."""
        if rng_key is None:
            raise ValueError("rng_key is required for sampling taus in DActorCritic")

        bs = state.shape[0]
        cos, taus = self.calc_cos(bs, rng_key)

        cos_embd = self.cosine_embedding(cos)
        state_embd = self.state_embedding(state)
        state_embd = state_embd.reshape(-1, 1, state_embd.shape[-1])

        comb_embd = (state_embd * cos_embd).reshape(bs * self.n_taus, -1)

        x = self.fc2(comb_embd)
        x = nnx.relu(x)
        out = self.fc3(x)

        return out.reshape(bs, self.n_taus, -1), jnp.squeeze(taus, -1)

    def calc_cos(self, batch_size, rng_key):
        """Calculate cosine values for quantile embedding."""
        pis = jnp.array([jnp.pi * i for i in range(1, self.n_cosines + 1)])
        pis = pis.reshape(1, 1, self.n_cosines)

        taus = (
            jax.random.uniform(
                rng_key,
                (batch_size, self.n_taus),
                minval=self.lb,
                maxval=1.0,
            )
            * self.ub
        )
        taus = jnp.expand_dims(taus, -1)

        cos = jnp.cos(taus * pis)
        assert cos.shape == (
            batch_size,
            self.n_taus,
            self.n_cosines,
        ), "cos shape is incorrect"
        return cos, taus


class SoftQNetwork(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        layer_sizes: tuple = (256, 256),
        activation: str = "relu",
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation
        in_dim = state_dim + action_dim
        _layers = []
        for layer_dim in layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, state, action):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = jnp.concatenate([state, action], axis=-1)
        for layer in self.layers:
            x = activation(layer(x))
        return self.output_layer(x)


class Actor(nnx.Module):
    """Standalone actor network for rollouts/evaluation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        env_action_dim: int = None,
        mpc_fn=None,
        layer_sizes: tuple = (256, 256),
        activation: str = "tanh",
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mpc_fn = mpc_fn
        self.env_action_dim = (
            env_action_dim if env_action_dim is not None else action_dim
        )
        self.activation_name = activation

        _layers = []
        in_dim = obs_dim
        for layer_dim in layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(self.env_action_dim))

    def __call__(self, x, physical_state=None):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.layers:
            x = activation(layer(x))
        out = self.actor_output(x)

        if self.mpc_fn is not None:
            actor_mean = self.mpc_fn(out, physical_state)
        else:
            actor_mean = out

        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(jnp.clip(self.actor_log_std.value, -5.0, 2.0))
        )


def get_sac_action_dist(pi, scale_action, bias_action, _rng, action, sample: bool):
    """
    Sample from a Gaussian policy pi, apply tanh squashing, and rescale to the
    environment action space with (scale_action, bias_action).
    """

    def sample_fn(action):
        return pi.sample(seed=_rng)

    def use_action_fn(action):
        pi.sample(seed=_rng)
        return action

    x_t = jax.lax.cond(sample, sample_fn, use_action_fn, action)

    y_t = jnp.tanh(x_t)
    log_prob = pi.log_prob(x_t) - jnp.log(scale_action * (1 - y_t**2) + 1e-6).sum(-1)
    mean_tanh = jnp.tanh(pi.mean())
    action_out = y_t * scale_action + bias_action
    mean = mean_tanh * scale_action + bias_action
    log_prob = jnp.expand_dims(log_prob, -1)
    return action_out, log_prob, mean, x_t


def get_ddpg_action_dist(pi, scale_action, bias_action, _rng):
    mean_tanh = jnp.tanh(pi.mean())
    mean = mean_tanh * scale_action + bias_action
    return mean, None, mean


class SpatialEncoding(nnx.Module):
    """Spatial/Positional encoding for transformer models."""

    def __init__(
        self,
        feature_dim: int,
        max_seq_len: int = 1024,
        rngs: nnx.Rngs = None,
    ):
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.div_term = jnp.exp(
            jnp.arange(0, feature_dim, 2) * -(jnp.log(10000.0) / feature_dim)
        )
        self.has_odd_features = feature_dim % 2 == 1

    def __call__(self, x):
        """
        Add positional encoding to input.
        Args:
            x: (batch_size, seq_len, feature_dim)
        Returns:
            (batch_size, seq_len, feature_dim)
        """
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        # Compute positional encoding on the fly
        position = jnp.arange(seq_len, dtype=jnp.float32)[:, None]  # (seq_len, 1)

        pe = jnp.zeros((seq_len, self.feature_dim), dtype=x.dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(position * self.div_term))
        if self.has_odd_features:
            pe = pe.at[:, 1::2].set(jnp.cos(position * self.div_term[:-1]))
        else:
            pe = pe.at[:, 1::2].set(jnp.cos(position * self.div_term))

        return (
            x + pe[None, :, :]
        )  # Broadcast to batch (batch_size, seq_len, feature_dim)


class TransformerEncoderBlock(nnx.Module):
    """Single transformer encoder block with multi-head attention and FFN."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        ffn_hidden_dim: int = 2048,
        activation: str = "relu",
        rngs: nnx.Rngs = None,
    ):
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.activation_name = activation

        # Multi-head attention
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=feature_dim,
            out_features=feature_dim,
            qkv_features=feature_dim,
            deterministic=True,
            decode=False,
            kernel_init=orthogonal(0.1),
            bias_init=constant(0.0),
            out_kernel_init=orthogonal(0.1),
            out_bias_init=constant(0.0),
            rngs=rngs,
        )
        self.mha_ln = nnx.LayerNorm(feature_dim, rngs=rngs)

        # Feed-forward network
        self.ffn_dense1 = nnx.Linear(
            feature_dim,
            ffn_hidden_dim,
            kernel_init=orthogonal(0.1),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.ffn_dense2 = nnx.Linear(
            ffn_hidden_dim,
            feature_dim,
            kernel_init=orthogonal(0.1),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.ffn_ln = nnx.LayerNorm(feature_dim, rngs=rngs)

    def __call__(self, x, mask=None, training: bool = False):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
            mask: Optional attention mask
            training: Whether in training mode (for dropout)
        Returns:
            (batch_size, seq_len, feature_dim)
        """
        # Multi-head attention with residual connection
        x = self.mha_ln(x)
        attn_out = self.mha(x, mask=mask, deterministic=not training)
        x = 0.5 * x + attn_out
        # x = x - jnp.mean(x, axis=-1, keepdims=True)/(jnp.std(x, axis=-1, keepdims=True) + 1e-8) # LayerNorm without learnable parameters

        # Feed-forward with residual connection
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        ffn_out = activation(self.ffn_dense1(x))
        ffn_out = self.ffn_dense2(ffn_out)
        x = 0.5 * x + ffn_out
        # x = x - jnp.mean(x, axis=-1, keepdims=True)/(jnp.std(x, axis=-1, keepdims=True) + 1e-8) # LayerNorm without learnable parameters
        return x


class TransformerActorCritic(nnx.Module):
    """Transformer-based Actor-Critic with spatial encoding."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        obs_seq_len: int,
        env_action_dim: int = None,
        mpc_fn=None,
        activation: str = "relu",
        transformer_hidden_dim: int = 128,
        num_heads: int = 2,
        num_encoder_layers: int = 1,
        ffn_hidden_dim: int = 128,
        max_seq_len: int = 1024,
        actor_head_hidden_dim: tuple[int, ...] = (128, 128),
        actor_seq_len: int | None = None,
        critic_layer_sizes: tuple[int, ...] = (256, 256),
        rngs: nnx.Rngs = None,
    ):
        self.action_dim = action_dim
        self.mpc_fn = mpc_fn
        self.env_action_dim = (
            env_action_dim if env_action_dim is not None else action_dim
        )
        self.activation_name = activation
        self.transformer_hidden_dim = transformer_hidden_dim
        self.obs_seq_len = obs_seq_len
        self.actor_seq_len = actor_seq_len if actor_seq_len is not None else obs_seq_len

        # Input projection to transformer hidden dim
        self.input_projection = nnx.Linear(
            obs_dim,
            transformer_hidden_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )

        # Spatial/Positional encoding
        self.spatial_encoding = SpatialEncoding(
            transformer_hidden_dim,
            max_seq_len=max_seq_len,
            rngs=rngs,
        )

        # Transformer encoder blocks
        encoder_blocks = []
        for _ in range(num_encoder_layers):
            encoder_blocks.append(
                TransformerEncoderBlock(
                    feature_dim=transformer_hidden_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    activation=activation,
                    rngs=rngs,
                )
            )
        self.encoder_blocks = nnx.List(encoder_blocks)

        # Build actor layers
        _actor_layers = []
        in_dim = transformer_hidden_dim * self.actor_seq_len
        for layer_dim in actor_head_hidden_dim:
            _actor_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.actor_hidden = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(
            in_dim,
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.actor_log_std = nnx.Param(jnp.zeros(self.env_action_dim))

        # Critic head: simple MLP on raw flattened observations (independent from transformer)
        # This allows the critic to learn value function without depending on transformer features
        # Build critic layers
        _critic_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def _encode(self, x, training: bool = False):
        """
        Encode observation through spatial encoding and transformer blocks.
        Args:
            x: (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
            training: Whether in training mode
        Returns:
            (batch_size, seq_len, transformer_hidden_dim) or (batch_size, transformer_hidden_dim)
        """
        # Handle both 2D and 3D inputs
        squeeze_output = False
        if x.ndim == 2:
            x = x[:, None, :]  # Add sequence dimension: (batch_size, 1, obs_dim)
            squeeze_output = True

        # Project to transformer hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, transformer_hidden_dim)
        x = nnx.tanh(x)
        # Add spatial encoding
        x = self.spatial_encoding(x)

        # Apply transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)

        # Remove sequence dimension if input was 2D
        if squeeze_output:
            x = x[:, 0, :]  # (batch_size, transformer_hidden_dim)

        return x

    def __call__(self, x, physical_state=None, training: bool = False):
        """Full forward pass with actor and critic."""
        pi = self.actor(x, physical_state, training=training)
        value = self.critic(x, training=training)
        return pi, value

    def actor(self, x, physical_state=None, training: bool = False):
        """Actor forward pass."""
        # Encode through transformer
        encoded = self._encode(x, training=training)
        if encoded.ndim == 3:
            encoded = encoded[
                :, -self.actor_seq_len :, :
            ]  # Use last actor_seq_len tokens

            # Flatten the sequence and feature dimensions
            batch_size = encoded.shape[0]
            encoded = encoded.reshape(batch_size, -1)

        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for hidden_layer in self.actor_hidden:
            encoded = activation(hidden_layer(encoded))
        weights_matrix = self.actor_output(encoded)

        if self.mpc_fn is not None:
            actor_mean = self.mpc_fn(weights_matrix, physical_state)
        else:
            actor_mean = weights_matrix

        log_std = jnp.clip(self.actor_log_std.value, -3.0, 2.0)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))

    def critic(self, x, training: bool = False):
        """Critic forward pass.

        Uses simple MLP on raw observations, independent from transformer.
        This allows better value function learning without coupling to actor's
        transformer representations.
        """
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = x[:, -1, :] if x.ndim == 3 else x  # Use raw obs for critic
        for layer in self.critic_layers:
            x = activation(layer(x))
        return jnp.squeeze(self.critic_output_layer(x), axis=-1)


class MAPPOActor(nnx.Module):
    """Decentralized MAPPO actor with weight-sharing across agents.

    Each agent gets its own local observation and produces its own action,
    but the MLP weights are shared across all agents.

    Input:  (batch, num_agents, local_obs_dim)
    Output: MultivariateNormalDiag with event shape (num_agents * action_dim_per_agent,)
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim_per_agent: int,
        num_agents: int,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.local_obs_dim = local_obs_dim
        self.action_dim_per_agent = action_dim_per_agent
        self.num_agents = num_agents
        self.activation_name = activation

        _layers = []
        in_dim = local_obs_dim
        for layer_dim in actor_layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim,
            action_dim_per_agent,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        # Per-agent, per-action-dim learnable log_std
        self.actor_log_std = nnx.Param(
            jnp.zeros((num_agents, action_dim_per_agent))
        )

    def __call__(self, local_obs):
        """Forward pass.

        Args:
            local_obs: (batch, num_agents, local_obs_dim)
        Returns:
            distrax.MultivariateNormalDiag over flattened (batch, num_agents * action_dim)
        """
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        batch_size = local_obs.shape[0]
        # Process each agent independently through the shared MLP
        x = local_obs.reshape(-1, self.local_obs_dim)
        for layer in self.layers:
            x = activation(layer(x))
        x = self.output_layer(x)
        # Reshape back to (batch, num_agents, action_dim_per_agent)
        actor_mean = x.reshape(batch_size, self.num_agents, self.action_dim_per_agent)

        log_std = jnp.clip(self.actor_log_std.value, -5.0, 2.0)
        std = jnp.exp(log_std)[None, :, :]
        std = jnp.broadcast_to(std, actor_mean.shape)

        return distrax.MultivariateNormalDiag(
            actor_mean,
            std,
        )


class MAPPOCritic(nnx.Module):
    """Centralized MAPPO critic.

    Takes the global state and outputs a single scalar value.

    Input:  (batch, global_state_dim)
    Output: (batch,)
    """

    def __init__(
        self,
        global_state_dim: int,
        *,
        activation: str = "tanh",
        critic_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.activation_name = activation

        _layers = []
        in_dim = global_state_dim
        for layer_dim in critic_layer_sizes:
            _layers.append(
                nnx.Linear(
                    in_dim,
                    layer_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    rngs=rngs,
                )
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, global_state):
        """Forward pass.

        Args:
            global_state: (batch, global_state_dim)
        Returns:
            (batch,) scalar value
        """
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = global_state
        for layer in self.layers:
            x = activation(layer(x))
        return jnp.squeeze(self.output_layer(x), axis=-1)

class MultiAgentActorCritic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        env_action_dim: int = None,
        mpc_fn=None,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        critic_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mpc_fn = mpc_fn
        self.env_action_dim = env_action_dim if env_action_dim is not None else action_dim
        self.activation_name = activation

        _actor_layers = []
        in_dim = obs_dim
        for layer_dim in actor_layer_sizes:
            _actor_layers.append(
                nnx.Linear(in_dim, layer_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), rngs=rngs)
            )
            in_dim = layer_dim
        self.actor_layers = nnx.List(_actor_layers)
        self.actor_output = nnx.Linear(in_dim, action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), rngs=rngs)
        self.actor_log_std = nnx.Param(jnp.zeros(self.env_action_dim))

        _critic_layers = []
        in_dim = obs_dim
        for layer_dim in critic_layer_sizes:
            _critic_layers.append(
                nnx.Linear(in_dim, layer_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), rngs=rngs)
            )
            in_dim = layer_dim
        self.critic_layers = nnx.List(_critic_layers)
        self.critic_output_layer = nnx.Linear(in_dim, 1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), rngs=rngs)

    def __call__(self, x, physical_state=None, reference=None):
        pi = self.actor(x, physical_state, reference)
        value = self.critic(x)
        return pi, value

    def actor(self, x, physical_state=None, reference=None):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = x[:, -1, :] if x.ndim == 3 else x
        for layer in self.actor_layers:
            x = activation(layer(x))

        out = self.actor_output(x)

        if self.mpc_fn is not None:
            actor_mean = self.mpc_fn(out, physical_state, reference)
        else:
            actor_mean = out

        log_std = jnp.clip(self.actor_log_std.value, -5.0, 2.0)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std))

    def critic(self, x):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = x[:, -1, :] if x.ndim == 3 else x
        for layer in self.critic_layers:
            x = activation(layer(x))
        return jnp.squeeze(self.critic_output_layer(x), axis=-1)

class MAPPODiffMPCActor(nnx.Module):
    def __init__(
        self,
        local_obs_dim: int,
        mpc_weight_dim: int,
        env_action_dim_per_agent: int,
        num_agents: int,
        mpc_fn,
        *,
        activation: str = "tanh",
        actor_layer_sizes: tuple = (256, 256),
        rngs: nnx.Rngs,
    ):
        self.local_obs_dim = local_obs_dim
        self.mpc_weight_dim = mpc_weight_dim
        self.env_action_dim_per_agent = env_action_dim_per_agent
        self.num_agents = num_agents
        self.activation_name = activation
        self.mpc_fn = mpc_fn

        _layers = []
        in_dim = local_obs_dim
        for layer_dim in actor_layer_sizes:
            _layers.append(
                nnx.Linear(in_dim, layer_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), rngs=rngs)
            )
            in_dim = layer_dim
        self.layers = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim, mpc_weight_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), rngs=rngs
        )
        self.actor_log_std = nnx.Param(jnp.zeros((num_agents, env_action_dim_per_agent)))

    def __call__(self, local_obs, physical_state=None, reference=None):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        x = local_obs[:, :, -1, :] if local_obs.ndim == 4 else local_obs
        
        # Share weights across agents by reshaping
        batch_size = x.shape[0]
        x = x.reshape(batch_size * self.num_agents, self.local_obs_dim)

        for layer in self.layers:
            x = activation(layer(x))

        out = self.output_layer(x)
        out = out.reshape(batch_size, self.num_agents, self.mpc_weight_dim)

        if self.mpc_fn is not None:
            # mpc_fn takes (weights, physical_state, reference)
            # physical_state: (batch, num_agents, 2)
            # reference: (batch, num_agents, 2)
            # out: (batch, num_agents, mpc_weight_dim)
            # We assume mpc_fn is vmapped over both batch and num_agents!
            actor_mean = self.mpc_fn(out, physical_state, reference)
        else:
            actor_mean = out

        actor_mean_flat = actor_mean.reshape(batch_size, self.num_agents * self.env_action_dim_per_agent)
        log_std = jnp.clip(self.actor_log_std.value, -5.0, 2.0)
        log_std_flat = log_std.reshape(-1)
        log_std_flat = jnp.broadcast_to(log_std_flat, actor_mean_flat.shape)

        return distrax.MultivariateNormalDiag(actor_mean_flat, jnp.exp(log_std_flat))


class MAPPOTransformerDiffMPCActor(nnx.Module):
    def __init__(
        self,
        local_obs_dim: int,
        mpc_weight_dim: int,
        env_action_dim_per_agent: int,
        num_agents: int,
        mpc_fn,
        *,
        obs_seq_len: int = 1,
        activation: str = "relu",
        transformer_hidden_dim: int = 128,
        num_heads: int = 2,
        num_encoder_layers: int = 1,
        ffn_hidden_dim: int = 128,
        actor_layer_sizes: tuple = (256, 256),
        max_seq_len: int = 1024,
        rngs: nnx.Rngs,
    ):
        self.local_obs_dim = local_obs_dim
        self.mpc_weight_dim = mpc_weight_dim
        self.env_action_dim_per_agent = env_action_dim_per_agent
        self.num_agents = num_agents
        self.activation_name = activation
        self.mpc_fn = mpc_fn
        self.transformer_hidden_dim = transformer_hidden_dim
        self.obs_seq_len = obs_seq_len
        
        self.input_projection = nnx.Linear(
            local_obs_dim,
            transformer_hidden_dim,
            kernel_init=orthogonal(0.1),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        
        self.spatial_encoding = SpatialEncoding(
            transformer_hidden_dim,
            max_seq_len=max_seq_len,
            rngs=rngs,
        )
        
        encoder_blocks = []
        for _ in range(num_encoder_layers):
            encoder_blocks.append(
                TransformerEncoderBlock(
                    feature_dim=transformer_hidden_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    activation=activation,
                    rngs=rngs,
                )
            )
        self.encoder_blocks = nnx.List(encoder_blocks)

        _layers = []
        in_dim = transformer_hidden_dim
        for layer_dim in actor_layer_sizes:
            _layers.append(
                nnx.Linear(in_dim, layer_dim, kernel_init=orthogonal(0.1), bias_init=constant(0.0), rngs=rngs)
            )
            in_dim = layer_dim
        self.actor_hidden = nnx.List(_layers)
        self.output_layer = nnx.Linear(
            in_dim, mpc_weight_dim, kernel_init=orthogonal(0.1), bias_init=constant(0.0), rngs=rngs
        )
        self.actor_log_std = nnx.Param(jnp.zeros((num_agents, env_action_dim_per_agent)))

    def _encode(self, x, training: bool = False):
        squeeze_output = False
        if x.ndim == 2: # (batch*num_agents, obs_dim)
            x = x[:, None, :] 
            squeeze_output = True
        
        x = self.input_projection(x)
        x = nnx.tanh(x)
        x = self.spatial_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
            
        if squeeze_output:
            x = x[:, 0, :]
            
        return x

    def __call__(self, local_obs, physical_state=None, reference=None, training: bool = False):
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        # local_obs could be (batch, num_agents, seq_len, obs_dim) or (batch, num_agents, obs_dim)
        if local_obs.ndim == 4:
            batch_size = local_obs.shape[0]
            x = local_obs.reshape(batch_size * self.num_agents, self.obs_seq_len, self.local_obs_dim)
        elif local_obs.ndim == 3:
            batch_size = local_obs.shape[0]
            x = local_obs.reshape(batch_size * self.num_agents, self.local_obs_dim)
        else:
            raise ValueError(f"Unexpected local_obs shape: {local_obs.shape}")

        encoded = self._encode(x, training=training)
        if encoded.ndim == 3:
            # Use the representation of the most recent observation (last token)
            encoded = encoded[:, -1, :]
            
        for layer in self.actor_hidden:
            encoded = activation(layer(encoded))

        out = self.output_layer(encoded)
        out = out.reshape(batch_size, self.num_agents, self.mpc_weight_dim)

        if self.mpc_fn is not None:
            actor_mean = self.mpc_fn(out, physical_state, reference)
        else:
            actor_mean = out

        actor_mean_flat = actor_mean.reshape(batch_size, self.num_agents * self.env_action_dim_per_agent)
        log_std = jnp.clip(self.actor_log_std.value, -5.0, 2.0)
        log_std_flat = log_std.reshape(-1)
        log_std_flat = jnp.broadcast_to(log_std_flat, actor_mean_flat.shape)

        return distrax.MultivariateNormalDiag(actor_mean_flat, jnp.exp(log_std_flat))
class MAPPOTransformerCritic(nnx.Module):
    """
    Centralized Critic using a Transformer to process the history sequence of all agents.
    """
    def __init__(
        self,
        local_obs_dim: int,
        num_agents: int,
        *,
        obs_seq_len: int = 1,
        activation: str = "relu",
        transformer_hidden_dim: int = 128,
        num_heads: int = 2,
        num_encoder_layers: int = 1,
        ffn_hidden_dim: int = 128,
        critic_layer_sizes: tuple = (256, 256),
        max_seq_len: int = 1024,
        rngs: nnx.Rngs,
    ):
        self.local_obs_dim = local_obs_dim
        self.num_agents = num_agents
        self.obs_seq_len = obs_seq_len
        self.activation_name = activation
        self.transformer_hidden_dim = transformer_hidden_dim

        # Input dimension for each token in the sequence: N * D_local + 1 (for time_frac)
        in_token_dim = num_agents * local_obs_dim + 1

        self.input_projection = nnx.Linear(
            in_token_dim,
            transformer_hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )

        self.spatial_encoding = SpatialEncoding(
            transformer_hidden_dim,
            max_seq_len=max_seq_len,
            rngs=rngs,
        )

        encoder_blocks = []
        for _ in range(num_encoder_layers):
            encoder_blocks.append(
                TransformerEncoderBlock(
                    feature_dim=transformer_hidden_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    activation=activation,
                    rngs=rngs,
                )
            )
        self.encoder_blocks = nnx.List(encoder_blocks)

        _layers = []
        in_dim = transformer_hidden_dim
        for layer_dim in critic_layer_sizes:
            _layers.append(
                nnx.Linear(in_dim, layer_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), rngs=rngs)
            )
            in_dim = layer_dim
        self.critic_hidden = nnx.List(_layers)
        
        self.output_layer = nnx.Linear(
            in_dim, 1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), rngs=rngs
        )

    def _encode(self, x, training: bool = False):
        x = self.input_projection(x)
        x = nnx.tanh(x)
        x = self.spatial_encoding(x)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
            
        return x

    def __call__(self, global_state, training: bool = False):
        """
        Args:
            global_state: (batch, N * seq_len * D_local + 1)
        Returns:
            (batch,)
        """
        batch_size = global_state.shape[0]
        
        time_frac = global_state[:, -1:]
        hist_flat = global_state[:, :-1]
        
        # reshape history to (batch, seq_len, N * D_local)
        hist_seq = hist_flat.reshape(batch_size, self.obs_seq_len, self.num_agents * self.local_obs_dim)
        
        time_frac_seq = jnp.broadcast_to(time_frac[:, None, :], (batch_size, self.obs_seq_len, 1))
        
        # (batch, seq_len, N * D_local + 1)
        seq_input = jnp.concatenate([hist_seq, time_frac_seq], axis=-1)
        
        encoded = self._encode(seq_input, training=training)
        
        # Take the last token
        if encoded.ndim == 3:
            encoded = encoded[:, -1, :]
            
        x = encoded
        activation = nnx.relu if self.activation_name == "relu" else nnx.tanh
        for layer in self.critic_hidden:
            x = activation(layer(x))
            
        out = self.output_layer(x)
        return jnp.squeeze(out, axis=-1)

