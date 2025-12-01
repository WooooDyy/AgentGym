# Migration Guide: v0.1.0 → v0.2.0

## What's New

AgentEnv MCP v0.2.0 introduces **MCP Client/Server architecture** with SSE (Server-Sent Events) communication using the `fastmcp` package.

### New Features

1. **MCP Client Mode**: Connect to external MCP servers via SSE
2. **FastMCP Integration**: Use fastmcp for unified client/server interaction
3. **Configuration-based Setup**: Use `mcp.json` to configure MCP servers
4. **Multi-server Support**: Connect to multiple MCP servers simultaneously
5. **Server Management**: Start/stop MCP servers with the `start_servers.py` script

## Breaking Changes

### File Structure Changes

**Before (v0.1.0)**:
```
examples/
├── __init__.py
└── directional.py
```

**After (v0.2.0)**:
```
examples/
├── __init__.py
├── start_servers.py          # NEW: Server launcher
└── directional/              # CHANGED: Now a subdirectory
    ├── __init__.py
    ├── server.py             # NEW: FastMCP server
    └── state.py              # MOVED: Was directional.py
```

### Import Changes

**Before**:
```python
from examples.directional import DirectionalToolSet, DirectionalState
```

**After**:
```python
# Still works! (backward compatible)
from examples.directional import DirectionalToolSet, DirectionalState

# Or access the new server
from examples.directional import server  # Available in server.py module
```

## Migration Steps

### Option 1: Continue Using Internal Mode (No Changes Required)

Your existing code continues to work:

```python
from agentenv_mcp import server
from examples.directional import DirectionalToolSet, DirectionalState

server.set_factories(DirectionalToolSet, DirectionalState)
```

### Option 2: Migrate to MCP Client Mode

#### Step 1: Install Dependencies

```bash
uv pip install -e .
```

This now includes `fastmcp>=0.2.0`.

#### Step 2: Create Configuration

Copy the example configuration:
```bash
cp agentenv_mcp/mcp.example.json agentenv_mcp/mcp.json
```

Customize as needed:
```json
{
  "mcpServers": {
    "directional": {
      "command": "uv",
      "args": ["run", "python", "-m", "examples.directional.server"],
      "url": "http://localhost:8001",
      "description": "Directional navigation MCP server"
    }
  },
  "defaultServer": "directional"
}
```

#### Step 3: Update Your Code

**Before**:
```python
from agentenv_mcp import server, app
from examples.directional import DirectionalToolSet, DirectionalState
import uvicorn

server.set_factories(DirectionalToolSet, DirectionalState)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**After**:
```python
from agentenv_mcp import server_v2, app
import uvicorn

# Configure MCP clients
server_v2.set_mcp_config("agentenv_mcp/mcp.json", default_client="directional")

# Start server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or use the new launcher:
```bash
mcp-client --config agentenv_mcp/mcp.json --port 8000
```

#### Step 4: Start MCP Servers

Before starting your environment server, start the MCP servers:

```bash
# Terminal 1: Start MCP server
python examples/start_servers.py directional

# Terminal 2: Start environment server
mcp-client --config agentenv_mcp/mcp.json
```

## New API Components

### Server Instances

- `server` (v0.1.0): Original internal mode server
- `server_v2` (v0.2.0): New MCP client mode server

### Environment Classes

- `MCPEnvironment` (v0.1.0): Original internal mode environment
- `MCPBasedEnvironment` (v0.2.0): New MCP client-based environment

### Client Classes

- `MCPClient`: Connect to a single MCP server
- `MCPClientManager`: Manage multiple MCP servers

## Command Line Tools

### New Scripts

1. **`mcp-client`**: Launch environment server in MCP client mode
   ```bash
   mcp-client --config mcp.json --port 8000
   ```

2. **`start_servers.py`**: Launch MCP servers
   ```bash
   python examples/start_servers.py directional
   python examples/start_servers.py --list
   ```

### Existing Scripts (Unchanged)

1. **`mcp`**: Launch environment server in internal mode
   ```bash
   mcp --example directional --port 8000
   ```

## Configuration File

### New: `mcp.json`

Location: `agentenv_mcp/mcp.json`

```json
{
  "mcpServers": {
    "server-name": {
      "command": "command-to-start-server",
      "args": ["arg1", "arg2"],
      "url": "http://localhost:PORT",
      "description": "Server description"
    }
  },
  "defaultServer": "server-name"
}
```

## Custom MCP Servers

### Before (v0.1.0): Internal Mode

Create `examples/myserver.py`:

```python
from agentenv_mcp.mcp_base import MCPState, MCPToolSet

class MyState(MCPState):
    def reset(self):
        self.data = {}

    def get_observation(self) -> str:
        return str(self.data)

class MyToolSet(MCPToolSet):
    def get_tools(self):
        return {"mytool": self.tool_mytool}

    def tool_mytool(self, state: MyState) -> str:
        return "Tool executed"

    def execute_tool(self, tool_name: str, state: MCPState, params=None) -> str:
        return self.get_tools()[tool_name](state)
```

### After (v0.2.0): FastMCP Server

Create `examples/myserver/server.py`:

```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

_state = {}

@mcp.tool()
def mytool() -> str:
    """Execute my tool."""
    return "Tool executed"

@mcp.tool()
def reset() -> str:
    """Reset state."""
    _state.clear()
    return "Reset complete"

if __name__ == "__main__":
    mcp.run(transport="sse")
```

## Testing

### Internal Mode (Original)

```bash
# Terminal 1
uv run python -m agentenv_mcp.launch --example directional --port 8004

# Terminal 2
uv run python test_env.py
```

### MCP Client Mode (New)

```bash
# Terminal 1: Start MCP server
python examples/start_servers.py directional

# Terminal 2: Start environment server
mcp-client --config agentenv_mcp/mcp.json --port 8000

# Terminal 3: Run tests
# (Update test_env.py to use port 8000)
uv run python test_env.py
```

## Benefits of Migrating

### Internal Mode
- ✅ Lower latency (no network overhead)
- ✅ Simpler setup
- ✅ Better for prototyping
- ❌ Limited scalability
- ❌ Tightly coupled

### MCP Client Mode
- ✅ Distributed architecture
- ✅ Multiple tool providers
- ✅ Configuration-based
- ✅ Better scalability
- ❌ Network overhead
- ❌ More complex setup

## Troubleshooting

### Import Errors

If you get import errors after updating:

```bash
# Reinstall package
uv pip install -e .
```

### MCP Server Connection Issues

Check that:
1. MCP server is running: `curl http://localhost:8001/sse`
2. Port is correct in `mcp.json`
3. No firewall blocking the connection

### FastMCP Not Found

```bash
# Install fastmcp
uv pip install fastmcp
```

## Support

For issues or questions:
- Check the updated [README.md](README.md)
- Review example implementations in `examples/directional/`
- Open an issue on GitHub

## Version Compatibility

- **v0.1.0**: Internal mode only
- **v0.2.0**: Both internal mode (backward compatible) and MCP client mode

You can run both modes side-by-side by using different ports.
