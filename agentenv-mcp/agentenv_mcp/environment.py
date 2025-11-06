"""
Generic MCP Environment - Works with any MCP tool implementation.

The environment is completely agnostic to what tools exist.
Tools are injected via MCPToolSet implementations.
"""

import threading
from typing import Dict, Any, Optional, List
from .mcp_base import MCPState, MCPToolSet
from .rewards import RewardCalculator


class MCPEnvironment:
    """
    Generic MCP environment instance.

    Works with any MCPToolSet and MCPState implementation.
    Actions = tool calls. Observations = tool results.
    """

    def __init__(
        self,
        toolset: MCPToolSet,
        state: MCPState,
        task_description: str = "Interact with MCP tools",
        reward_calculator: Optional[RewardCalculator] = None,
        max_steps: int = 50
    ):
        """
        Initialize environment instance.

        Args:
            toolset: MCP tool set implementation
            state: MCP state implementation
            task_description: Description of the task
            reward_calculator: Optional reward calculator
            max_steps: Maximum steps per episode
        """
        self.toolset = toolset
        self.state = state
        self.task_description = task_description
        self.reward_calculator = reward_calculator
        self.max_steps = max_steps

        # Episode state
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self) -> str:
        """Reset environment to initial state."""
        self.state.reset()
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False

        if self.reward_calculator:
            self.reward_calculator.reset()

        # Return initial observation
        obs = f"Task: {self.task_description}\n\n"
        obs += f"{self.state.get_observation()}\n\n"
        obs += f"Available actions: {self.toolset.get_action_space()}"

        return obs

    def step(self, action: str) -> Dict[str, Any]:
        """
        Execute an action (tool call).

        Args:
            action: Tool name to execute

        Returns:
            Dict with observation, reward, score, done
        """
        if self.done:
            return {
                "observation": "Episode is done. Please reset.",
                "reward": 0.0,
                "score": self.total_reward,
                "done": True
            }

        self.current_step += 1

        # Execute tool on state
        observation = self.toolset.execute_tool(action, self.state)

        # Calculate reward
        reward = 0.1  # Default small reward for valid action

        if self.reward_calculator:
            reward = self.reward_calculator.update(observation, action, {})
            if self.reward_calculator.is_goal_reached():
                self.done = True

        self.total_reward += reward

        # Check max steps
        if self.current_step >= self.max_steps:
            self.done = True

        return {
            "observation": observation,
            "reward": reward,
            "score": self.total_reward,
            "done": self.done
        }

    def observe(self) -> str:
        """Get current observation without taking action."""
        return self.state.get_observation()

    def get_action_space(self) -> List[str]:
        """Get available actions."""
        return self.toolset.get_action_space()


class MCPEnvServer:
    """
    Generic multi-instance environment manager.

    Manages multiple MCPEnvironment instances with pluggable tool sets.
    """

    def __init__(self, toolset_factory=None, state_factory=None):
        """
        Initialize server.

        Args:
            toolset_factory: Factory function to create MCPToolSet instances
            state_factory: Factory function to create MCPState instances
        """
        self._max_id = 0
        self.envs: Dict[int, MCPEnvironment] = {}
        self.info: Dict[int, Dict[str, Any]] = {}
        self.ls: list = []
        self._lock = threading.Lock()

        # Factories for creating tool sets and states
        self.toolset_factory = toolset_factory
        self.state_factory = state_factory

    def set_factories(self, toolset_factory, state_factory):
        """
        Set factories for creating tool sets and states.

        Args:
            toolset_factory: Callable that returns MCPToolSet instance
            state_factory: Callable that returns MCPState instance
        """
        self.toolset_factory = toolset_factory
        self.state_factory = state_factory

    def create(self) -> Dict[str, int]:
        """Create a new environment instance."""
        try:
            with self._lock:
                idx = self._max_id
                self._max_id += 1

            self.info[idx] = {"deleted": False, "done": False}
            self.ls.append(idx)

            return {"id": idx}
        except Exception as e:
            return {"error": str(e)}

    def reset(self, env_id: int, data_idx: int = 0) -> Dict[str, Any]:
        """
        Reset environment instance.

        Args:
            env_id: Environment ID
            data_idx: Task/data index

        Returns:
            Dict with observation, reward, score, done
        """
        try:
            if not self.toolset_factory or not self.state_factory:
                return {"error": "Toolset and state factories not configured. Call set_factories() first."}

            # Create environment if it doesn't exist
            if env_id not in self.envs:
                toolset = self.toolset_factory()
                state = self.state_factory()
                task = f"Complete task {data_idx}"

                self.envs[env_id] = MCPEnvironment(
                    toolset=toolset,
                    state=state,
                    task_description=task,
                    reward_calculator=None,
                    max_steps=50
                )

            # Reset the environment
            observation = self.envs[env_id].reset()

            payload = {
                "observation": observation,
                "reward": 0.0,
                "score": 0.0,
                "done": False,
                "deleted": False
            }

            self.info[env_id].update(payload)
            return payload

        except Exception as e:
            return {"error": str(e)}

    def step(self, env_id: int, action: str) -> Dict[str, Any]:
        """
        Execute an action in the environment.

        Args:
            env_id: Environment ID
            action: Action to take (tool name)

        Returns:
            Dict with observation, reward, score, done
        """
        try:
            if env_id not in self.envs:
                return {"error": f"Environment {env_id} not found"}

            if self.info[env_id]["deleted"]:
                return {"error": f"Environment {env_id} has been deleted"}

            if self.info[env_id]["done"]:
                return {"error": f"Environment {env_id} is done. Please reset."}

            # Execute step
            result = self.envs[env_id].step(action)

            # Update info
            self.info[env_id].update(result)
            self.info[env_id]["deleted"] = False

            return result

        except Exception as e:
            return {"error": str(e)}

    def observe(self, env_id: int) -> Dict[str, Any]:
        """Get current observation without taking action."""
        try:
            if env_id not in self.envs:
                return {"error": f"Environment {env_id} not found"}

            observation = self.envs[env_id].observe()

            return {
                "observation": observation,
                "reward": self.envs[env_id].total_reward,
                "score": self.envs[env_id].total_reward,
                "done": self.envs[env_id].done,
                "deleted": self.info[env_id].get("deleted", False)
            }

        except Exception as e:
            return {"error": str(e)}

    def close(self, env_id: int) -> bool:
        """Close and cleanup environment."""
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)

            if env_id in self.envs:
                del self.envs[env_id]

            if env_id in self.info:
                self.info[env_id]["deleted"] = True

            print(f"Environment {env_id} closed")
            return True

        except Exception as e:
            print(f"Error closing environment {env_id}: {e}")
            return False

    def __del__(self):
        """Cleanup all environments on deletion."""
        for idx in list(self.ls):
            try:
                self.close(idx)
            except:
                pass


# Global server instance
server = MCPEnvServer()
