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
        # x = self.spatial_encoding(x)

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
