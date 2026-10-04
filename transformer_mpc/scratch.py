import jax
import jax.numpy as jnp
from flax import nnx

from networks import TransformerActorCritic, MAPPOTransformerDiffMPCActor

rng = jax.random.PRNGKey(0)

print("Testing TransformerActorCritic...")
net1 = TransformerActorCritic(
    obs_dim=10, action_dim=10, env_action_dim=4, obs_seq_len=1, 
    rngs=nnx.Rngs(rng)
)
g1, s1 = nnx.split(net1)
p1 = s1.filter(nnx.Param)
np1 = s1.filter(lambda path, v: not isinstance(v, nnx.Param))
print("s1 leaves:", len(jax.tree_util.tree_leaves(s1)))
print("p1 leaves:", len(jax.tree_util.tree_leaves(p1)))
print("np1 leaves:", len(jax.tree_util.tree_leaves(np1)))
print("np1 dict:", list(np1.keys()) if isinstance(np1, dict) else np1)

print("\nTesting MAPPOTransformerDiffMPCActor...")
net2 = MAPPOTransformerDiffMPCActor(
    local_obs_dim=10, mpc_weight_dim=10, env_action_dim_per_agent=4, num_agents=2,
    mpc_fn=None, rngs=nnx.Rngs(rng)
)
g2, s2 = nnx.split(net2)
p2 = s2.filter(nnx.Param)
np2 = s2.filter(lambda path, v: not isinstance(v, nnx.Param))
print("s2 leaves:", len(jax.tree_util.tree_leaves(s2)))
print("p2 leaves:", len(jax.tree_util.tree_leaves(p2)))
print("np2 leaves:", len(jax.tree_util.tree_leaves(np2)))
print("np2 dict:", list(np2.keys()) if isinstance(np2, dict) else np2)
