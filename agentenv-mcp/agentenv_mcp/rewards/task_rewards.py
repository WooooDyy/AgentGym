"""
Task-specific reward calculators for MCP environment.
"""

import re
from typing import Dict, Any, List, Optional, Callable
from .base import RewardCalculator


class ToolCallReward(RewardCalculator):
    """
    Reward calculator based on correct tool call sequence.

    Similar to BabyAI's action sequence rewards, this rewards the agent
    for making the correct sequence of tool calls.
    """

    def __init__(
        self,
        expected_tools: List[str],
        max_reward: float = 1.0,
        partial_credit: bool = True,
    ):
        """
        Initialize tool call reward calculator.

        Args:
            expected_tools: List of expected tool names in order
            max_reward: Maximum reward for completing all tool calls
            partial_credit: Whether to give partial credit for progress
        """
        super().__init__(max_reward)
        self.expected_tools = expected_tools
        self.partial_credit = partial_credit
        self.completed_tools: List[str] = []
        self.num_expected = len(expected_tools)

    def calculate(self, observation: str, action: str, info: Dict[str, Any]) -> float:
        """Calculate reward based on tool call progress."""
        # Extract tool name from action or observation
        tool_name = self._extract_tool_name(action)

        if not tool_name:
            return self.current_reward

        # Check if this is the next expected tool
        if len(self.completed_tools) < self.num_expected:
            next_expected = self.expected_tools[len(self.completed_tools)]
            if tool_name == next_expected:
                self.completed_tools.append(tool_name)

                if self.partial_credit:
                    # Give partial reward for each correct tool
                    return (
                        len(self.completed_tools) / self.num_expected
                    ) * self.max_reward
                elif len(self.completed_tools) == self.num_expected:
                    # Only give reward when all tools are used
                    return self.max_reward

        return self.current_reward

    def is_goal_reached(self) -> bool:
        """Check if all expected tools have been called."""
        return len(self.completed_tools) == self.num_expected

    def reset(self):
        """Reset tool call tracking."""
        super().reset()
        self.completed_tools = []

    def _extract_tool_name(self, action: str) -> Optional[str]:
        """Extract tool name from action string."""
        # Try to match "Action: tool_name" pattern
        match = re.search(r"Action:\s*(\w+)", action, re.IGNORECASE)
        if match:
            return match.group(1)
        return None


class QueryAnswerReward(RewardCalculator):
    """
    Reward calculator for query -> tool call -> answer workflow.

    Rewards the agent for:
    1. Using appropriate tools to gather information
    2. Providing correct final answer
    """

    def __init__(
        self,
        required_info: List[str],
        answer_checker: Optional[Callable[[str], bool]] = None,
        max_reward: float = 1.0,
    ):
        """
        Initialize query-answer reward calculator.

        Args:
            required_info: List of required information patterns to gather
            answer_checker: Optional function to validate final answer
            max_reward: Maximum reward
        """
        super().__init__(max_reward)
        self.required_info = required_info
        self.answer_checker = answer_checker
        self.gathered_info: List[str] = []
        self.answer_provided = False

    def calculate(self, observation: str, action: str, info: Dict[str, Any]) -> float:
        """Calculate reward based on information gathering and answer quality."""
        # Check for information gathering
        for pattern in self.required_info:
            if pattern not in self.gathered_info and self._matches_pattern(
                observation, pattern
            ):
                self.gathered_info.append(pattern)

        # Check for final answer
        if self._is_finish_action(action):
            self.answer_provided = True
            answer = self._extract_answer(action)

            # Calculate reward based on gathered info and answer quality
            info_reward = (len(self.gathered_info) / len(self.required_info)) * 0.5

            if self.answer_checker and answer:
                answer_reward = 0.5 if self.answer_checker(answer) else 0.0
            else:
                # If no checker provided, reward for providing any answer
                answer_reward = 0.5 if answer else 0.0

            return (info_reward + answer_reward) * self.max_reward

        # Partial reward for gathering information
        return (
            (len(self.gathered_info) / len(self.required_info)) * 0.5 * self.max_reward
        )

    def is_goal_reached(self) -> bool:
        """Check if all information gathered and answer provided."""
        return (
            len(self.gathered_info) == len(self.required_info) and self.answer_provided
        )

    def reset(self):
        """Reset information gathering and answer tracking."""
        super().reset()
        self.gathered_info = []
        self.answer_provided = False

    def _matches_pattern(self, observation: str, pattern: str) -> bool:
        """Check if observation matches required information pattern."""
        return pattern.lower() in observation.lower()

    def _is_finish_action(self, action: str) -> bool:
        """Check if action is a finish/answer action."""
        return bool(re.search(r"Action:\s*finish", action, re.IGNORECASE))

    def _extract_answer(self, action: str) -> Optional[str]:
        """Extract answer from finish action."""
        match = re.search(r"answer[:\s]+(.+)", action, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


class GoalBasedReward(RewardCalculator):
    """
    Flexible goal-based reward calculator.

    Allows custom goal checking function and milestone tracking.
    """

    def __init__(
        self,
        goal_checker: Callable[[List[str], Dict[str, Any]], float],
        milestones: Optional[List[Callable[[str, Dict[str, Any]], bool]]] = None,
        max_reward: float = 1.0,
    ):
        """
        Initialize goal-based reward calculator.

        Args:
            goal_checker: Function that takes observation history and info,
                         returns reward value (0.0 to 1.0)
            milestones: Optional list of milestone checking functions
            max_reward: Maximum reward
        """
        super().__init__(max_reward)
        self.goal_checker = goal_checker
        self.milestones = milestones or []
        self.completed_milestones: List[int] = []

    def calculate(self, observation: str, action: str, info: Dict[str, Any]) -> float:
        """Calculate reward using custom goal checker."""
        # Check milestones
        for i, milestone_fn in enumerate(self.milestones):
            if i not in self.completed_milestones:
                if milestone_fn(observation, info):
                    self.completed_milestones.append(i)

        # Calculate overall progress using goal checker
        progress = self.goal_checker(self.observation_history, info)
        return progress * self.max_reward

    def is_goal_reached(self) -> bool:
        """Check if goal is reached using goal checker."""
        progress = self.goal_checker(self.observation_history, {})
        return progress >= 1.0

    def reset(self):
        """Reset milestone tracking."""
        super().reset()
        self.completed_milestones = []


def create_reward_calculator(
    task_type: str, task_config: Dict[str, Any], max_reward: float = 1.0
) -> RewardCalculator:
    """
    Factory function to create reward calculator based on task type.

    Args:
        task_type: Type of task ("tool_sequence", "query_answer", "custom")
        task_config: Configuration for the reward calculator
        max_reward: Maximum reward

    Returns:
        Appropriate RewardCalculator instance
    """
    if task_type == "tool_sequence":
        return ToolCallReward(
            expected_tools=task_config.get("expected_tools", []),
            max_reward=max_reward,
            partial_credit=task_config.get("partial_credit", True),
        )

    elif task_type == "query_answer":
        return QueryAnswerReward(
            required_info=task_config.get("required_info", []),
            answer_checker=task_config.get("answer_checker"),
            max_reward=max_reward,
        )

    elif task_type == "custom":
        return GoalBasedReward(
            goal_checker=task_config["goal_checker"],
            milestones=task_config.get("milestones"),
            max_reward=max_reward,
        )

    else:
        raise ValueError(f"Unknown task type: {task_type}")
