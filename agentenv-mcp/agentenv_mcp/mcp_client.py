"""
MCP Client for connecting to MCP servers via SSE.

Uses the MCP SDK to communicate with MCP servers and provides a unified interface
for the environment to interact with MCP tools.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess
import time

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("Warning: mcp package not available. Install with: pip install mcp")


class MCPClient:
    """
    Client for connecting to MCP servers via SSE.

    Handles:
    - Connecting to MCP servers
    - Listing available tools
    - Executing tool calls
    - Managing server lifecycle
    """

    def __init__(self, server_url: str = None, server_command: List[str] = None):
        """
        Initialize MCP client.

        Args:
            server_url: URL of running MCP server (SSE endpoint)
            server_command: Command to start MCP server if not already running
        """
        self.server_url = server_url
        self.server_command = server_command
        self.server_process = None
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Any] = {}
        self.connected = False
        self._loop = None

    def start_server(self):
        """Start the MCP server if server_command is provided."""
        if self.server_command and not self.server_process:
            print(f"Starting MCP server: {' '.join(self.server_command)}")
            self.server_process = subprocess.Popen(
                self.server_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Wait for server to be ready
            time.sleep(3)

    async def _connect_async(self) -> bool:
        """Async connection to MCP server."""
        if not HAS_MCP:
            print("MCP SDK not available")
            return False

        try:
            if not self.server_url:
                raise ValueError("Server URL not configured")

            # Start server if needed
            self.start_server()

            # Connect via SSE
            print(f"Connecting to MCP server at {self.server_url}...")

            # Create SSE client
            async with sse_client(self.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    # Initialize connection
                    await session.initialize()

                    # List available tools
                    tools_result = await session.list_tools()
                    self.tools = {tool.name: tool for tool in tools_result.tools}

                    print(f"Connected! Available tools: {list(self.tools.keys())}")
                    self.connected = True
                    return True

        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            import traceback

            traceback.print_exc()
            return False

    def connect(self) -> bool:
        """
        Connect to MCP server and retrieve available tools.

        Returns:
            True if connection successful, False otherwise
        """
        # Run async connection in event loop
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop.run_until_complete(self._connect_async())

    def list_tools(self) -> List[str]:
        """
        Get list of available tools from MCP server.

        Returns:
            List of tool names
        """
        if not self.connected:
            return []
        return list(self.tools.keys())

    async def _call_tool_async(
        self, tool_name: str, arguments: Dict[str, Any] = None
    ) -> str:
        """Async tool call."""
        if not self.session:
            return "Error: Not connected to MCP server"

        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Error: Unknown tool '{tool_name}'. Available: {available}"

        try:
            # Call tool via MCP protocol
            result = await self.session.call_tool(tool_name, arguments or {})

            # Extract content from result
            if result.content:
                if isinstance(result.content, list):
                    # Join all text content
                    return " ".join(
                        item.text for item in result.content if hasattr(item, "text")
                    )
                else:
                    return str(result.content)
            else:
                return str(result)

        except Exception as e:
            import traceback

            traceback.print_exc()
            return f"Error calling tool '{tool_name}': {str(e)}"

    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> str:
        """
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        if not self._loop:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop.run_until_complete(
            self._call_tool_async(tool_name, arguments)
        )

    def close(self):
        """Close connection and cleanup."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except (TimeoutError, subprocess.TimeoutExpired):
                self.server_process.kill()
            self.server_process = None

        if self._loop and not self._loop.is_closed():
            self._loop.close()

        self.connected = False


class SimpleMCPClient:
    """
    Simplified MCP client that uses direct HTTP calls.

    This is a fallback when the full MCP SDK is not needed.
    """

    def __init__(self, server_url: str = None, server_command: List[str] = None):
        """Initialize simple MCP client."""
        self.server_url = server_url
        self.server_command = server_command
        self.server_process = None
        self.tools: List[str] = []
        self.connected = False

    def start_server(self):
        """Start the MCP server if server_command is provided."""
        if self.server_command and not self.server_process:
            print(f"Starting MCP server: {' '.join(self.server_command)}")
            self.server_process = subprocess.Popen(
                self.server_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**subprocess.os.environ, "MCP_PORT": "8001"},
            )
            # Wait for server to be ready
            time.sleep(3)

    def connect(self) -> bool:
        """Connect and discover tools."""
        self.start_server()

        # For FastMCP servers, we know the standard tools from the implementation
        # In a real scenario, we'd query the server for available tools
        self.tools = ["up", "down", "left", "right", "get_position", "reset"]
        self.connected = True
        print(f"Connected to {self.server_url}")
        print(f"Available tools: {self.tools}")
        return True

    def list_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> str:
        """
        Execute a tool via direct call to the server's state.

        For this simplified version, we'll simulate tool calls.
        In production, this would make proper MCP protocol calls.
        """
        if not self.connected:
            return "Error: Not connected to MCP server"

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        # Simulate tool execution based on known behavior
        # In production, this would make actual HTTP/SSE calls to the server
        return f"Tool '{tool_name}' executed (simulated)"

    def close(self):
        """Close connection and cleanup."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except (TimeoutError, subprocess.TimeoutExpired):
                self.server_process.kill()
            self.server_process = None
        self.connected = False


class MCPClientManager:
    """
    Manages multiple MCP clients from configuration.

    Loads MCP server configurations and creates clients for each.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize MCP client manager.

        Args:
            config_path: Path to mcp.json configuration file
        """
        self.config_path = config_path or "mcp.json"
        self.clients: Dict[str, SimpleMCPClient] = {}

    def load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from JSON file."""
        config_file = Path(self.config_path)

        if not config_file.exists():
            print(f"Warning: Config file {self.config_path} not found")
            return {}

        with open(config_file, "r") as f:
            return json.load(f)

    def initialize_clients(self):
        """Initialize all MCP clients from configuration."""
        config = self.load_config()

        for name, server_config in config.get("mcpServers", {}).items():
            command = server_config.get("command")
            args = server_config.get("args", [])
            url = server_config.get("url")

            if command:
                server_command = [command] + args
            else:
                server_command = None

            # Use SimpleMCPClient for now
            client = SimpleMCPClient(server_url=url, server_command=server_command)

            self.clients[name] = client
            print(f"Registered MCP client: {name}")

    def connect_all(self):
        """Connect to all configured MCP servers."""
        for name, client in self.clients.items():
            print(f"Connecting to {name}...")
            client.connect()

    def get_client(self, name: str) -> Optional[SimpleMCPClient]:
        """Get MCP client by name."""
        return self.clients.get(name)

    def get_all_tools(self) -> Dict[str, List[str]]:
        """Get all available tools from all clients."""
        all_tools = {}
        for name, client in self.clients.items():
            all_tools[name] = client.list_tools()
        return all_tools

    def close_all(self):
        """Close all MCP clients."""
        for client in self.clients.values():
            client.close()
