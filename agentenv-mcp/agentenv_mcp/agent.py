"""
Agent for interacting with MCP environments.

Provides a simple agent that can explore MCP servers by calling tools.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from .client import MCPClient
from .logging_config import get_logger

logger = get_logger("agent")


@dataclass
class AgentTrajectory:
    """Records agent's trajectory through the environment."""

    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add_step(self, action: str, observation: str, reward: float = 0.0) -> None:
        """Add a step to the trajectory."""
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)

    def total_reward(self) -> float:
        """Get total accumulated reward."""
        return sum(self.rewards)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actions": self.actions,
            "observations": self.observations,
            "rewards": self.rewards,
            "total_reward": self.total_reward(),
        }


class MCPAgent:
    """
    Agent that interacts with MCP servers.

    Uses FastMCP client to discover and execute tools.
    """

    def __init__(self, server_url: str, max_steps: int = 50):
        """
        Initialize the agent.

        Args:
            server_url: URL of MCP server to connect to
            max_steps: Maximum steps per episode
        """
        self.client = MCPClient(server_url)
        self.max_steps = max_steps
        self.trajectory = AgentTrajectory()
        self._tools: list[str] = []
        logger.info(f"Created MCPAgent for {server_url}")

    async def discover_tools(self) -> list[str]:
        """
        Discover available tools from the MCP server.

        Returns:
            List of available tool names
        """
        self._tools = await self.client.list_tools()
        logger.info(f"Discovered tools: {self._tools}")
        return self._tools

    async def step(self, action: str) -> str:
        """
        Execute an action (tool call) on the MCP server.

        Args:
            action: Tool name to execute

        Returns:
            Observation from the tool
        """
        observation = await self.client.call_tool(action)
        reward = 0.1 if not observation.startswith("Error") else -0.1
        self.trajectory.add_step(action, observation, reward)
        logger.debug(f"Step: {action} -> {observation[:50]}...")
        return observation

    async def reset(self) -> str:
        """
        Reset the environment.

        Returns:
            Initial observation
        """
        self.trajectory = AgentTrajectory()
        if "reset" in self._tools:
            return await self.client.call_tool("reset")
        return "Environment ready"

    async def run_episode(self, policy: str = "random") -> AgentTrajectory:
        """
        Run a complete episode with the given policy.

        Args:
            policy: Policy to use ("random" for random actions)

        Returns:
            Complete trajectory
        """
        await self.discover_tools()
        await self.reset()

        # Filter out reset tool from action space
        action_space = [t for t in self._tools if t != "reset"]

        if not action_space:
            logger.warning("No actions available")
            return self.trajectory

        logger.info(f"Starting episode with {len(action_space)} available actions")

        for step in range(self.max_steps):
            if policy == "random":
                action = random.choice(action_space)
            else:
                action = action_space[0]  # Default: first action

            observation = await self.step(action)
            logger.info(f"Step {step + 1}: {action} -> {observation}")

            # Simple termination check
            if "done" in observation.lower() or "goal" in observation.lower():
                logger.info("Episode completed (goal reached)")
                break

        return self.trajectory


async def run_agent_demo(server_url: str, num_steps: int = 10) -> dict[str, Any]:
    """
    Run a demo of the agent interacting with an MCP server.

    Args:
        server_url: URL of MCP server
        num_steps: Number of steps to run

    Returns:
        Trajectory data
    """
    agent = MCPAgent(server_url, max_steps=num_steps)
    trajectory = await agent.run_episode(policy="random")

    result = {
        "trajectory": trajectory.to_dict(),
        "tools": agent._tools,
        "server_url": server_url,
    }

    logger.info(f"Demo complete. Total reward: {trajectory.total_reward()}")
    return result
