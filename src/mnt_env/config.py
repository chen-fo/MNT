"""Explicit, serializable task parameters; no algorithm dependencies."""

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class MntConfig:
    size: int = 16
    num_allies: int = 2
    num_enemies: int = 2
    num_mines: int = 8
    sonar_range: int = 8
    shoot_range: int = 3
    max_steps: int = 100
    target_reward: float = 50.0
    gamma: float = 0.95
    failure_penalty: float = 10.0

    def __post_init__(self):
        for key in ("size", "num_allies", "num_enemies", "num_mines", "sonar_range",
                    "shoot_range", "max_steps"):
            value = getattr(self, key)
            minimum = 0 if key == "num_mines" else 5 if key == "size" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.num_allies + self.num_enemies + self.num_mines + 1 > self.size ** 2:
            raise ValueError("Not enough cells for the default non-overlapping initial layout")
        for key in ("target_reward", "gamma", "failure_penalty"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite and numeric")
        if self.target_reward < 0 or self.failure_penalty <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("Require target_reward >= 0, failure_penalty > 0 and 0 < gamma <= 1")
