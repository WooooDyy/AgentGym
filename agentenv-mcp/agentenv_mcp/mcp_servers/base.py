"""
Base classes for MCP server implementations.

Provides abstract base classes for creating extensible MCP servers with FastMCP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP

from ..logging_config import get_logger

logger = get_logger("mcp_servers.base")


@dataclass
class MCPServerState:
    """
    Base state class for MCP servers.

    Subclass to define environment-specific state.
    """

    step_count: int = 0
    history: list = field(default_factory=list)

    def reset(self) -> None:
        """Reset state to initial values."""
        self.step_count = 0
        self.history = []

    def get_observation(self) -> str:
        """Get current observation string."""
        return f"Step: {self.step_count}"

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {"step_count": self.step_count, "history": self.history}


class BaseMCPServer(ABC):
    """
    Abstract base class for MCP servers.

    Provides a template for creating FastMCP servers with tools.
    Subclass and implement register_tools() to add environment-specific tools.
    """

    def __init__(self, name: str, state: MCPServerState | None = None):
        """
        Initialize the MCP server.

        Args:
            name: Server name for FastMCP
            state: Initial state (creates default if None)
        """
        self.name = name
        self.mcp = FastMCP(name)
        self.state = state or self._create_default_state()
        self._register_tools()
        logger.info(f"Initialized MCP server: {name}")

    @abstractmethod
    def _create_default_state(self) -> MCPServerState:
        """Create the default state for this server type."""
        pass

    @abstractmethod
    def _register_tools(self) -> None:
        """Register tools with the FastMCP server."""
        pass

    def reset(self) -> str:
        """Reset the environment state."""
        self.state.reset()
        logger.info(f"Server {self.name} reset")
        return self.state.get_observation()

    def get_observation(self) -> str:
        """Get current observation."""
        return self.state.get_observation()

    def run(self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8001):
        """
        Run the MCP server.

        Args:
            transport: Transport type ("sse" or "stdio")
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"Starting {self.name} on {host}:{port} ({transport})")
        self.mcp.run(transport=transport, host=host, port=port)
