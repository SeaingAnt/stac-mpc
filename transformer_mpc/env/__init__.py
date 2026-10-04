from .drone_v0 import DroneV0
from .drone_v1 import DroneV1
from .drone_v2 import DroneV2
from .multi_agent_nav import MultiAgentNav2D
from .multi_agent_nav_v2 import MultiAgentNav2DV2

ENV_REGISTRY = {
    "drone_v0": DroneV0,
    "drone_v1": DroneV1,
    "drone_v2": DroneV2,
    "multi_agent_nav": MultiAgentNav2D,
    "multi_agent_nav_v2": MultiAgentNav2DV2,
}

__all__ = [
    "DroneV0",
    "DroneV1",
    "DroneV2",
    "MultiAgentNav2D",
    "MultiAgentNav2DV2",
]
