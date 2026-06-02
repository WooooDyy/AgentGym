"""
FastMCP Client for connecting to MCP servers.

Uses FastMCP Client to connect to MCP servers and provides
a unified interface for the agent to interact with MCP tools.
"""

import asyncio
from typing import Any

from fastmcp import Client

from .logging_config import get_logger

logger = get_logger("client")


class MCPClient:
    """
    Client for connecting to MCP servers using FastMCP.

    Handles connection management and tool execution.
    """

    def __init__(self, server_url: str):
        """
        Initialize MCP client.

        Args:
            server_url: URL of MCP server (e.g., "http://localhost:8001/sse")
        """
        self.server_url = server_url
        self.client = Client(server_url)
        self._tools: dict[str, Any] = {}
        self._connected = False
        logger.info(f"Created MCPClient for {server_url}")

    async def connect(self) -> bool:
        """
        Connect to the MCP server and discover available tools.

        Returns:
            True if connection successful
        """
        try:
            async with self.client:
                tools = await self.client.list_tools()
                self._tools = {tool.name: tool for tool in tools}
                self._connected = True
                logger.info(f"Connected. Available tools: {list(self._tools.keys())}")
                return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def list_tools(self) -> list[str]:
        """
        Get list of available tools from the MCP server.

        Returns:
            List of tool names
        """
        async with self.client:
            tools = await self.client.list_tools()
            return [tool.name for tool in tools]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        try:
            async with self.client:
                result = await self.client.call_tool(tool_name, arguments or {})
                # Extract text content from result
                if hasattr(result, "content") and result.content:
                    if isinstance(result.content, list):
                        return " ".join(
                            getattr(item, "text", str(item)) for item in result.content
                        )
                    return str(result.content)
                return str(result)
        except Exception as e:
            logger.error(f"Tool call failed: {tool_name} - {e}")
            return f"Error: {e}"

    def call_tool_sync(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """
        Synchronous wrapper for call_tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        return asyncio.run(self.call_tool(tool_name, arguments))

    def list_tools_sync(self) -> list[str]:
        """
        Synchronous wrapper for list_tools.

        Returns:
            List of tool names
        """
        return asyncio.run(self.list_tools())
