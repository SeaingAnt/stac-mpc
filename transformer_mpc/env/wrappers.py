import jax
import jax.numpy as jnp
import chex
import numpy as np
from typing import NamedTuple, Optional, Tuple, Union
from gymnax.environments import environment, spaces
from utils import normalize


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(self._env.observation_space(params), spaces.Box), (
            "Only Box spaces are supported for now."
        )
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


class LogEnvState(NamedTuple):
    env_state: environment.EnvState
    episode_returns_true: float
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_returns_true: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0.0, 0, 0.0, 0.0, 0, 0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return_true = state.episode_returns_true + info.get("trw", reward)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns_true=new_episode_return_true * (1 - done),
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns_true=state.returned_episode_returns_true * (1 - done)
            + new_episode_return_true * done,
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns_true"] = state.returned_episode_returns_true
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(
            action,
            self._env.action_space(params).low,
            self._env.action_space(params).high,
        )
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


class NormalizeVecRewEnvState(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class MinMaxObservationWrapper(GymnaxWrapper):
    """Normalize the observations of the environment to [-1, 1]."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        super().__init__(env)
        self._obs_min = jnp.array(self._env.observation_space(params).low)
        self._obs_max = jnp.array(self._env.observation_space(params).high)
        assert jnp.isinf(self._obs_max).sum() == 0, "Obs space has infinities"
        assert jnp.isinf(self._obs_min).sum() == 0, "Obs space has infinities"

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        obs, env_state = self._env.reset(key, params)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return obs, env_state, reward, done, info


class ResetEnvWrapper(GymnaxWrapper):
    """Call reset when done is True."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        obs_st, state_st, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs_re, state_re = self.reset(key, params)

        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)

        info["real_next_obs"] = obs_st
        return obs, state, reward, done, info


class DomainRandomizationEnvState(NamedTuple):
    env_state: environment.EnvState
    params: environment.EnvParams


class DomainRandomizationWrapper(GymnaxWrapper):
    """Sample parameters at reset and auto-reset done environments."""

    def __init__(self, env, params: Optional[environment.EnvParams] = None):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        min_params = self._env.params_min
        max_params = self._env.params_max

        key, sample_key = jax.random.split(key)
        randomized_params = self.sample_params(sample_key, params, min_params, max_params)

        obs, env_state = self._env.reset(key, randomized_params)
        state = DomainRandomizationEnvState(env_state=env_state, params=randomized_params)
        return obs, state

    def sample_params(self, key, params, low_list, high_list):
        leaves, treedef = jax.tree_util.tree_flatten(params)
        assert len(leaves) == len(low_list) == len(high_list), (
            f"Mismatch: Params has {len(leaves)} nodes, but low/high lists have {len(low_list)}"
        )

        keys = jax.random.split(key, len(leaves))
        new_leaves = []
        for rng, leaf, l, h in zip(keys, leaves, low_list, high_list):
            sampled_val = jax.random.uniform(
                rng, shape=jnp.shape(leaf), minval=l, maxval=h
            )
            new_leaves.append(sampled_val)

        new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        return new_params

    def step(
        self,
        key: chex.PRNGKey,
        state: DomainRandomizationEnvState,
        action: Union[int, float, jax.Array],
        params: Optional[environment.EnvParams] = None,
    ):
        obs_st, env_state_st, reward, done, info = self._env.step(
            key, state.env_state, action, state.params
        )

        state_st = DomainRandomizationEnvState(env_state=env_state_st, params=state.params)
        obs_re, state_re = self.reset(key, params)

        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)

        info["real_next_obs"] = obs_st
        return obs, state, reward, done, info


class EvalVideoWrapper(GymnaxWrapper):
    """Render evaluation rollouts by delegating rendering to the base env."""

    def __init__(self, env):
        super().__init__(env)

    def _base_env(self):
        env = self._env
        while hasattr(env, "_env"):
            env = env._env
        return env

    @staticmethod
    def _unwrap_state(state):
        unwrapped = state
        while hasattr(unwrapped, "env_state"):
            unwrapped = unwrapped.env_state
        return unwrapped

    @staticmethod
    def _select_first_env(state):
        def _maybe_first(x):
            if isinstance(x, (jnp.ndarray, np.ndarray)) and x.ndim > 0:
                return x[0]
            return x

        return jax.tree.map(_maybe_first, state)

    def render_video(
        self,
        env_state_batch,
        output_path,
        max_frames,
        width,
        height,
        fps=None,
    ):
        base_env = self._base_env()
        if not hasattr(base_env, "render_video_from_states"):
            return None

        states = []
        for frame_idx in range(max_frames):
            state_t = jax.tree.map(lambda x, i=frame_idx: x[i], env_state_batch)
            state_t = self._unwrap_state(state_t)
            state_t = self._select_first_env(state_t)
            states.append(state_t)

        try:
            return base_env.render_video_from_states(
                states=states,
                output_path=output_path,
                fps=fps,
                width=width,
                height=height,
            )
        except Exception as exc:
            print(f"[render] EvalVideoWrapper failed; skipping video: {exc}")
            return None

    def get_render_fps(self, default_fps=30):
        base_env = self._base_env()
        if hasattr(base_env, "default_params") and hasattr(base_env, "mj_model"):
            dt = float(base_env.mj_model.opt.timestep) * float(base_env.default_params.frame_skip)
            return max(1, int(round(1.0 / dt)))
        return int(default_fps)
