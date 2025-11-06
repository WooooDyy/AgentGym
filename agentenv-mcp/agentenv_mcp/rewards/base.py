"""
Base reward calculator for MCP environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class RewardCalculator(ABC):
    """
    Abstract base class for reward calculation.

    Similar to BabyAI's obs_to_reward pattern, this allows flexible
    reward shaping based on observation history and expected outcomes.
    """

    def __init__(self, max_reward: float = 1.0):
        """
        Initialize reward calculator.

        Args:
            max_reward: Maximum possible reward for completing the task
        """
        self.max_reward = max_reward
        self.current_reward = 0.0
        self.reward_history: List[float] = []
        self.observation_history: List[str] = []

    @abstractmethod
    def calculate(
        self,
        observation: str,
        action: str,
        info: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for current step.

        Args:
            observation: Current observation string
            action: Action taken
            info: Additional information from environment

        Returns:
            Reward value for this step
        """
        pass

    @abstractmethod
    def is_goal_reached(self) -> bool:
        """
        Check if the goal has been reached.

        Returns:
            True if goal is achieved, False otherwise
        """
        pass

    def update(
        self,
        observation: str,
        action: str,
        info: Dict[str, Any]
    ) -> float:
        """
        Update reward state and return reward for this step.

        Args:
            observation: Current observation string
            action: Action taken
            info: Additional information from environment

        Returns:
            Reward value for this step
        """
        self.observation_history.append(observation)
        reward = self.calculate(observation, action, info)
        self.current_reward = max(self.current_reward, reward)
        self.reward_history.append(reward)
        return reward

    def reset(self):
        """Reset reward calculator to initial state."""
        self.current_reward = 0.0
        self.reward_history = []
        self.observation_history = []

    def get_progress(self) -> float:
        """
        Get current progress towards goal.

        Returns:
            Progress as fraction of max_reward (0.0 to 1.0)
        """
        return self.current_reward / self.max_reward if self.max_reward > 0 else 0.0
