"""
MCP Environment wrapper for RL training.

Provides a Gym-like interface for interacting with MCP servers.
"""

import asyncio
from typing import Any

from .client import MCPClient
from .logging_config import get_logger

logger = get_logger("environment")


class MCPEnvironment:
    """
    Gym-like environment wrapper for MCP servers.

    Uses FastMCP client to interact with MCP servers.
    Actions = tool calls, Observations = tool results.
    """

    def __init__(
        self,
        server_url: str,
        task_description: str = "Interact with MCP tools",
        max_steps: int = 50,
    ):
        """
        Initialize the MCP environment.

        Args:
            server_url: URL of the MCP server
            task_description: Description of the task
            max_steps: Maximum steps per episode
        """
        self.server_url = server_url
        self.client = MCPClient(server_url)
        self.task_description = task_description
        self.max_steps = max_steps

        self._tools: list[str] = []
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._history: list[dict[str, Any]] = []

        logger.info(f"Created MCPEnvironment for {server_url}")

    async def _discover_tools(self) -> list[str]:
        """Discover available tools from server."""
        self._tools = await self.client.list_tools()
        return self._tools

    @property
    def action_space(self) -> list[str]:
        """Get available actions (tools)."""
        return self._tools

    async def reset(self) -> str:
        """
        Reset the environment.

        Returns:
            Initial observation
        """
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._history = []

        # Discover tools
        await self._discover_tools()

        # Call reset tool if available
        if "reset" in self._tools:
            await self.client.call_tool("reset")

        obs = f"Task: {self.task_description}\n"
        obs += f"Available actions: {self._tools}"

        logger.info(f"Environment reset. Tools: {self._tools}")
        return obs

    async def step(self, action: str) -> dict[str, Any]:
        """
        Execute an action in the environment.

        Args:
            action: Tool name to execute

        Returns:
            Dict with observation, reward, done, info
        """
        if self._done:
            return {
                "observation": "Episode complete. Call reset().",
                "reward": 0.0,
                "done": True,
                "info": {"step": self._step_count},
            }

        self._step_count += 1

        # Execute the tool
        observation = await self.client.call_tool(action)

        # Calculate reward
        reward = 0.1 if not observation.startswith("Error") else -0.1
        self._total_reward += reward

        # Check termination
        if self._step_count >= self.max_steps:
            self._done = True

        # Record history
        self._history.append(
            {
                "step": self._step_count,
                "action": action,
                "observation": observation,
                "reward": reward,
            }
        )

        logger.debug(f"Step {self._step_count}: {action} -> reward={reward}")

        return {
            "observation": observation,
            "reward": reward,
            "done": self._done,
            "info": {
                "step": self._step_count,
                "total_reward": self._total_reward,
            },
        }

    def reset_sync(self) -> str:
        """Synchronous reset."""
        return asyncio.run(self.reset())

    def step_sync(self, action: str) -> dict[str, Any]:
        """Synchronous step."""
        return asyncio.run(self.step(action))
