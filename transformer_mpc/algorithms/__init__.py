"""Algorithm registry for unified training interface."""

from .base import AlgorithmSpec
from . import ppo
from . import diffmpc
from . import diffmpc_transformer
from . import diffmpc_transformer_stab

ALGORITHM_REGISTRY = {
    "ppo": ppo.SPEC,
    "diffmpc": diffmpc.SPEC,
    "diffmpc_transformer": diffmpc_transformer.SPEC,
    "diffmpc_transformer_stab": diffmpc_transformer_stab.SPEC,
}

__all__ = [
    "AlgorithmSpec",
    "ALGORITHM_REGISTRY",
]
