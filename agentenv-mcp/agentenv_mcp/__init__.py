"""
AgentEnv MCP - Model Context Protocol integration for AgentGym.

This package provides MCP server implementations and a FastMCP client
for building RL environments with tool-based action spaces.

Key Components:
- mcp_servers: Extensible MCP server implementations
- client: FastMCP client for connecting to MCP servers
- agent: Agent for interacting with MCP environments
- environment: Gym-like environment wrapper
"""

__version__ = "0.1.0"

from .agent import AgentTrajectory, MCPAgent
from .client import MCPClient
from .environment import MCPEnvironment
from .logging_config import get_logger, setup_logging

__all__ = [
    "MCPClient",
    "MCPAgent",
    "AgentTrajectory",
    "MCPEnvironment",
    "setup_logging",
    "get_logger",
]
