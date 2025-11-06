"""
Base classes for MCP tool implementations.

Provides abstract base for creating MCP servers with tools and state.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class MCPState(ABC):
    """
    Abstract base class for MCP environment state.

    Subclass this to define your environment's state.
    """

    @abstractmethod
    def reset(self):
        """Reset state to initial values."""
        pass

    @abstractmethod
    def get_observation(self) -> str:
        """Get current observation string."""
        pass


class MCPToolSet(ABC):
    """
    Abstract base class for MCP tool sets.

    Subclass this to define your environment's tools.
    """

    @abstractmethod
    def get_tools(self) -> Dict[str, callable]:
        """
        Get dictionary of available tools.

        Returns:
            Dict mapping tool names to callable functions
        """
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, state: MCPState, params: Dict[str, Any] = None) -> str:
        """
        Execute a tool on the given state.

        Args:
            tool_name: Name of tool to execute
            state: State to operate on
            params: Tool parameters

        Returns:
            Observation string
        """
        pass

    def get_action_space(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.get_tools().keys())
