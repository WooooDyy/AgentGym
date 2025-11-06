"""
Reward calculation module for MCP environment.

This module provides modular reward calculation based on observations,
similar to the obs_to_reward pattern in BabyAI.
"""

from .base import RewardCalculator
from .task_rewards import (
    ToolCallReward,
    QueryAnswerReward,
    GoalBasedReward,
    create_reward_calculator
)

__all__ = [
    "RewardCalculator",
    "ToolCallReward",
    "QueryAnswerReward",
    "GoalBasedReward",
    "create_reward_calculator",
]
