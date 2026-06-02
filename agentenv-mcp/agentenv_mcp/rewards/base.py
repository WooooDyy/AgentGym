"""
Base reward calculator classes.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..logging_config import get_logger

logger = get_logger("rewards")


class RewardCalculator(ABC):
    """Abstract base class for reward calculators."""

    @abstractmethod
    def calculate(self, action: str, observation: str, state: dict[str, Any]) -> float:
        """
        Calculate reward for a step.

        Args:
            action: Action taken
            observation: Resulting observation
            state: Current environment state

        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset calculator state."""
        pass


class DefaultRewardCalculator(RewardCalculator):
    """
    Default reward calculator.

    Provides small positive reward for valid actions, negative for errors.
    """

    def __init__(self, step_reward: float = 0.1, error_penalty: float = -0.1):
        self.step_reward = step_reward
        self.error_penalty = error_penalty
        self._total = 0.0

    def calculate(self, action: str, observation: str, state: dict[str, Any]) -> float:
        """Calculate reward based on observation."""
        if observation.startswith("Error"):
            reward = self.error_penalty
        else:
            reward = self.step_reward

        self._total += reward
        return reward

    def reset(self) -> None:
        """Reset accumulated reward."""
        self._total = 0.0

    @property
    def total_reward(self) -> float:
        """Get total accumulated reward."""
        return self._total
