# AgentEnv MCP - Generic MCP Environment Framework

Generic AgentGym environment with **two operational modes**:
1. **Internal Mode**: Tools define the action space with shared global state (original)
2. **MCP Client Mode**: Connect to external MCP servers via SSE using fastmcp (new)

## 🎯 Architecture

### New: MCP Client/Server Architecture (SSE-based)

```
┌─────────────────────────────────────────────┐
│  AgentGym Environment (agentenv_mcp/)       │
│  - MCPBasedEnvironment (uses MCP clients)   │
│  - MCPClient (SSE connection to servers)    │
│  - MCPClientManager (multi-server support)  │
└──────────────────┬──────────────────────────┘
                   │ connects via SSE
┌──────────────────▼──────────────────────────┐
│  MCP Servers (examples/*/server.py)         │
│  - FastMCP-based servers                    │
│  - Directional (up, down, left, right)      │
│  - Custom servers (add your own)            │
└─────────────────────────────────────────────┘
```

### Original: Internal Mode Architecture

```
┌─────────────────────────────────────────────┐
│  Generic MCP Environment (agentenv_mcp/)    │
│  - MCPEnvServer (multi-instance manager)    │
│  - MCPEnvironment (single instance)         │
│  - MCPState & MCPToolSet (base classes)     │
└──────────────────┬──────────────────────────┘
                   │ plugs in
┌──────────────────▼──────────────────────────┐
│  MCP Implementation (examples/*/state.py)   │
│  - DirectionalToolSet (up, down, left, right)│
│  - DirectionalState (position: {x, y})      │
└─────────────────────────────────────────────┘
```

## 📦 Installation

```bash
uv pip install -e .
```

This installs fastmcp and all required dependencies.

## 🚀 Quick Start

### Mode 1: MCP Client/Server (Recommended)

#### 1. Configure MCP Servers

Copy the example configuration:
```bash
cp agentenv_mcp/mcp.example.json agentenv_mcp/mcp.json
```

Edit `mcp.json` to configure your MCP servers:
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

#### 2. Start MCP Server(s)

```bash
# Start the directional MCP server
python examples/start_servers.py directional

# Or list available servers
python examples/start_servers.py --list
```

The server runs on `http://localhost:8001` with SSE transport.

#### 3. Use the Environment

**Via Python:**
```python
from agentenv_mcp import server_v2, app
import uvicorn

# Configure MCP clients
server_v2.set_mcp_config("agentenv_mcp/mcp.json", default_client="directional")

# Start AgentGym API server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Via HTTP API:**
```python
import requests

base_url = "http://localhost:8000"

# Create environment
resp = requests.post(f"{base_url}/create")
env_id = resp.json()["id"]

# Reset
resp = requests.post(f"{base_url}/reset", json={"id": env_id, "data_idx": 0})
print(resp.json()["observation"])

# Take action (call MCP tool)
resp = requests.post(f"{base_url}/step", json={"id": env_id, "action": "right"})
print(resp.json()["observation"])
```

### Mode 2: Internal Mode (Backward Compatible)

#### 1. Start the Server

```bash
uv run python -m agentenv_mcp.launch --example directional
```

#### 2. Test It

```bash
uv run python test_env.py
```

Expected output:
```
✓ Created environment
✓ Actions work: right → position (1, 0)
✓ Actions work: up → position (1, 1)
✓ Multiple instances maintain separate states
```

## 🔧 Creating Your Own MCP Server

### Step 1: Create Server Directory

```bash
mkdir -p examples/myserver
```

### Step 2: Create MCP Server with FastMCP

Create `examples/myserver/server.py`:

```python
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("My Custom Server")

# Server state
_state = {"value": 0}

@mcp.tool()
def increment() -> str:
    """Increment the counter."""
    _state["value"] += 1
    return f"Incremented to {_state['value']}"

@mcp.tool()
def decrement() -> str:
    """Decrement the counter."""
    _state["value"] -= 1
    return f"Decremented to {_state['value']}"

@mcp.tool()
def get_value() -> str:
    """Get current value."""
    return f"Current value: {_state['value']}"

@mcp.tool()
def reset() -> str:
    """Reset the counter."""
    _state["value"] = 0
    return "Counter reset to 0"

if __name__ == "__main__":
    mcp.run(transport="sse")
```

### Step 3: Register Server

Add to `examples/start_servers.py`:

```python
AVAILABLE_SERVERS = {
    "myserver": {
        "module": "examples.myserver.server",
        "port": 8002,
        "description": "My custom MCP server"
    },
    # ... other servers
}
```

### Step 4: Add to Configuration

Add to `mcp.json`:

```json
{
  "mcpServers": {
    "myserver": {
      "command": "uv",
      "args": ["run", "python", "-m", "examples.myserver.server"],
      "url": "http://localhost:8002",
      "description": "My custom MCP server"
    }
  }
}
```

### Step 5: Start and Use

```bash
# Start your server
python examples/start_servers.py myserver

# Use it in your environment by updating mcp.json defaultServer
```

## 📁 File Structure

```
agentenv-mcp/
├── agentenv_mcp/              # Core framework
│   ├── __init__.py            # Exports
│   ├── mcp_base.py            # MCPState, MCPToolSet (internal mode)
│   ├── environment.py         # Generic environment (internal mode)
│   ├── mcp_client.py          # MCP client (SSE connection)
│   ├── mcp_environment.py     # MCP-based environment (client mode)
│   ├── server.py              # AgentGym API endpoints
│   ├── model.py               # Pydantic models
│   ├── launch.py              # Server launcher
│   ├── mcp.example.json       # Example MCP configuration
│   └── rewards/               # Optional reward calculators
├── examples/                  # MCP server implementations
│   ├── start_servers.py       # Start script for MCP servers
│   ├── directional/           # Directional navigation example
│   │   ├── server.py          # FastMCP server
│   │   └── state.py           # State classes (internal mode)
│   └── __init__.py
├── test_env.py                # Test script
└── pyproject.toml             # uv dependencies
```

## 🎯 Example: Directional Navigation

**State**: Agent position on 2D grid
**Tools**: `up`, `down`, `left`, `right`, `get_position`, `reset`
**Goal**: Navigate to target positions

**Files**:
- MCP Server: `examples/directional/server.py`
- Internal Mode: `examples/directional/state.py`

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/health` | GET | Health check |
| `/create` | POST | Create environment instance |
| `/reset` | POST | Reset environment (loads state & tools) |
| `/step` | POST | Execute action (call tool) |
| `/observation` | GET | Get current observation |
| `/close` | POST | Close environment |

## 🧪 Testing

### Test MCP Server Mode

```bash
# Terminal 1: Start MCP server
python examples/start_servers.py directional

# Terminal 2: Start AgentGym environment server
# (TODO: Create launcher for MCP mode)

# Terminal 3: Run tests
uv run python test_env.py
```

### Test Internal Mode

```bash
# Terminal 1: Start server
uv run python -m agentenv_mcp.launch --example directional --port 8004

# Terminal 2: Run tests
uv run python test_env.py
```

## 🔑 Key Design Features

### MCP Client Mode
1. **SSE Communication**: FastMCP provides SSE (Server-Sent Events) transport
2. **External Servers**: MCP servers run as separate processes
3. **Configuration-based**: Use `mcp.json` to configure servers
4. **Multi-server Support**: Connect to multiple MCP servers
5. **Auto-start**: Can automatically start servers from configuration

### Internal Mode (Original)
1. **Generic Core**: Environment doesn't know what tools exist
2. **Pluggable Tools**: Tools injected via factory functions
3. **Shared State**: Tools operate on shared state per instance
4. **Observations from Tools**: Each tool returns observation after modifying state
5. **Multi-Instance**: Each environment has separate state

## 🆚 Comparison

| Feature | Internal Mode | MCP Client Mode |
|---------|--------------|-----------------|
| Tool execution | In-process | External MCP server |
| Communication | Direct function calls | SSE (HTTP) |
| State management | Per-instance classes | Server-managed |
| Scalability | Limited to process | Distributed |
| Flexibility | Requires code changes | Configuration-based |
| Performance | Faster (no network) | Network overhead |

## 📖 Use Cases

- **Internal Mode**: Fast prototyping, single-process setups, low latency
- **MCP Client Mode**: Production deployments, distributed systems, multiple tool providers

## 📄 License

MIT License
