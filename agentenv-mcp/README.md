# AgentEnv MCP

MCP (Model Context Protocol) integration for AgentGym environments using FastMCP.

## Overview

This package provides a modular framework for creating RL environments that expose their action space as MCP tools. Agents connect via FastMCP client to discover and execute tools.

### Architecture

```text
┌─────────────────┐     MCP/SSE      ┌─────────────────┐
│   MCPAgent      │◄────────────────►│   MCP Server    │
│  (FastMCP       │                  │  (FastMCP)      │
│   Client)       │                  │                 │
│                 │  list_tools()    │  @mcp.tool()    │
│                 │  call_tool()     │  - up()         │
│                 │                  │  - down()       │
│                 │                  │  - left()       │
│                 │                  │  - right()      │
└─────────────────┘                  └─────────────────┘
```

## Installation

```bash
cd agentenv-mcp
uv sync
```

## Quick Start

### 1. Start the MCP Server

```bash
# Run directional navigation server
uv run python -m agentenv_mcp.mcp_servers.directional

# Or via CLI
uv run agentenv-mcp server --port 8001
```

### 2. Run an Agent

```bash
# Run agent demo
uv run agentenv-mcp agent --server-url http://localhost:8001/sse --steps 10
```

### 3. End-to-End Test

```bash
uv run python test_e2e.py
```

## Project Structure

```text
agentenv-mcp/
├── agentenv_mcp/
│   ├── __init__.py          # Package exports
│   ├── client.py            # FastMCP client wrapper
│   ├── agent.py             # Agent for MCP interaction
│   ├── environment.py       # Gym-like environment wrapper
│   ├── logging_config.py    # Centralized logging
│   ├── cli.py               # CLI commands
│   ├── mcp_servers/         # MCP server implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base server classes
│   │   └── directional.py   # Directional navigation example
│   └── rewards/             # Reward calculators
│       ├── __init__.py
│       └── base.py
├── test_e2e.py              # End-to-end test
├── pyproject.toml
└── README.md
```

## Creating Custom MCP Servers

Extend `BaseMCPServer` to create new environments:

```python
from agentenv_mcp.mcp_servers.base import BaseMCPServer, MCPServerState
from dataclasses import dataclass

@dataclass
class MyState(MCPServerState):
    value: int = 0

class MyServer(BaseMCPServer):
    def _create_default_state(self) -> MyState:
        return MyState()

    def _register_tools(self) -> None:
        @self.mcp.tool()
        def increment() -> str:
            self.state.value += 1
            return f"Value: {self.state.value}"

        @self.mcp.tool()
        def decrement() -> str:
            self.state.value -= 1
            return f"Value: {self.state.value}"
```

## Using the Agent

```python
import asyncio
from agentenv_mcp import MCPAgent

async def main():
    agent = MCPAgent("http://localhost:8001/sse", max_steps=50)
    
    # Discover tools
    tools = await agent.discover_tools()
    print(f"Available tools: {tools}")
    
    # Run episode
    trajectory = await agent.run_episode(policy="random")
    print(f"Total reward: {trajectory.total_reward()}")

asyncio.run(main())
```

## Dependencies

- **fastmcp**: FastMCP library for MCP server and client
- **fastapi**: API framework (for future HTTP endpoints)
- **uvicorn**: ASGI server
- **pydantic**: Data validation

## License

MIT
