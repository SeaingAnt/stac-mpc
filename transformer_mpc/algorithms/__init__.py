"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import diffmpc
from . import multi_agent_diffmpc
from . import diffmpc_transformer
from . import diffmpc_transformer_stab
from . import mappo
from . import multi_agent_diffmpc_transformer
from . import multi_agent_diffmpc_transformer_stab

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "diffmpc": diffmpc.SPEC,
    "multi_agent_diffmpc": multi_agent_diffmpc.SPEC,
    "diffmpc_transformer": diffmpc_transformer.SPEC,
    "diffmpc_transformer_stab": diffmpc_transformer_stab.SPEC,
    "mappo": mappo.SPEC,
    "multi_agent_diffmpc_transformer": multi_agent_diffmpc_transformer.SPEC,
    "multi_agent_diffmpc_transformer_stab": multi_agent_diffmpc_transformer_stab.SPEC,
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
