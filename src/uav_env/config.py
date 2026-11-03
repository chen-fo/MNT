"""Serializable parameters for the UAV pursuit and evacuation task."""

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class UAVConfig:
    size: int = 16
    num_evaders: int = 2
    num_pursuers: int = 2
    vision_radius: int = 4
    max_steps: int = 100
    target_reward: float = 10.0
    failure_penalty: float = 10.0
    capture_reward: float = 10.0

    def __post_init__(self):
        for key in ("size", "num_evaders", "num_pursuers", "vision_radius", "max_steps"):
            value = getattr(self, key)
            minimum = 3 if key == "size" else 0 if key == "vision_radius" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
            object.__setattr__(self, key, int(value))
        if self.num_evaders + self.num_pursuers + 1 > self.size ** 2:
            raise ValueError("Not enough cells for agents and an unoccupied target")
        for key in ("target_reward", "failure_penalty", "capture_reward"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
            object.__setattr__(self, key, float(value))
        if self.target_reward < 0 or self.capture_reward < 0 or self.failure_penalty <= 0:
            raise ValueError("Require target_reward >= 0, capture_reward >= 0 and failure_penalty > 0")
