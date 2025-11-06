# AgentEnv MCP - Generic MCP Environment Framework

Generic AgentGym environment where **tools define the action space** and **share a global state**. Completely agnostic to what tools you implement.

## 🎯 Architecture

```
┌─────────────────────────────────────────────┐
│  Generic MCP Environment (agentenv_mcp/)    │  ← Core framework (tool-agnostic)
│  - MCPEnvServer (multi-instance manager)    │
│  - MCPEnvironment (single instance)         │
│  - MCPState & MCPToolSet (base classes)     │
└──────────────────┬──────────────────────────┘
                   │ plugs in
┌──────────────────▼──────────────────────────┐
│  Example MCP Implementation (examples/)     │  ← Specific tool implementations
│  - DirectionalToolSet (up, down, left, right)│
│  - DirectionalState (position: {x, y})      │
└─────────────────────────────────────────────┘
```

**Key Principle**: The core environment doesn't know what tools exist. Tools are injected via factory functions.

## 📦 Installation

```bash
uv pip install -e .
```

## 🚀 Quick Start

### 1. Start the Server (with directional example)

```bash
uv run python -m agentenv_mcp.launch --example directional
```

This loads the directional navigation example and starts the AgentGym API server.

### 2. Test It

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

## 💻 Usage

### Via HTTP API

```python
import requests

base_url = "http://localhost:8000"

# Create environment
resp = requests.post(f"{base_url}/create")
env_id = resp.json()["id"]

# Reset
resp = requests.post(f"{base_url}/reset", json={"id": env_id, "data_idx": 0})
print(resp.json()["observation"])
# "Task: Complete task 0
#  Current position: (0, 0). Steps taken: 0
#  Available actions: ['up', 'down', 'left', 'right', 'get_position']"

# Take action (call tool)
resp = requests.post(f"{base_url}/step", json={"id": env_id, "action": "right"})
print(resp.json()["observation"])
# "Moved right. Current position: (1, 0). Steps taken: 1"

# Close
requests.post(f"{base_url}/close", json={"id": env_id})
```

### Via AgentGym Client

```python
from agentenv.controller import Agent

agent = Agent(
    env_name="mcp",
    env_server_base="http://localhost:8000",
    data_len=10
)

obs = agent.reset(data_idx=0)
result = agent.step("right")  # Call the "right" tool
print(result["observation"])  # "Moved right. Current position: (1, 0)..."
```

## 🔧 Creating Your Own MCP Implementation

### Step 1: Define Your State

```python
from agentenv_mcp.mcp_base import MCPState

class MyState(MCPState):
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {"value": 0}

    def get_observation(self) -> str:
        return f"Current value: {self.data['value']}"
```

### Step 2: Define Your Tools

```python
from agentenv_mcp.mcp_base import MCPToolSet

class MyToolSet(MCPToolSet):
    def get_tools(self):
        return {
            "increment": self.tool_increment,
            "decrement": self.tool_decrement,
        }

    def tool_increment(self, state: MyState) -> str:
        state.data["value"] += 1
        return f"Incremented. {state.get_observation()}"

    def tool_decrement(self, state: MyState) -> str:
        state.data["value"] -= 1
        return f"Decremented. {state.get_observation()}"

    def execute_tool(self, tool_name: str, state: MCPState, params=None) -> str:
        tools = self.get_tools()
        if tool_name not in tools:
            return f"Error: Unknown tool '{tool_name}'"
        return tools[tool_name](state)
```

### Step 3: Configure and Run

```python
from agentenv_mcp import server, app

# Configure server with your tools
server.set_factories(MyToolSet, MyState)

# Start server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Or create `examples/my_implementation.py` and:

```bash
uv run python -m agentenv_mcp.launch --example my_implementation
```

## 📁 File Structure

```
agentenv-mcp/
├── agentenv_mcp/              # Generic core (tool-agnostic)
│   ├── __init__.py            # Exports
│   ├── mcp_base.py            # MCPState, MCPToolSet base classes
│   ├── environment.py         # Generic environment manager
│   ├── server.py              # AgentGym API endpoints
│   ├── model.py               # Pydantic models
│   ├── launch.py              # Server launcher
│   └── rewards/               # Optional reward calculators
├── examples/                  # Example MCP implementations
│   ├── directional.py         # Directional navigation example
│   └── __init__.py
├── test_env.py                # Test script
└── pyproject.toml             # uv dependencies
```

## 🎯 Example: Directional Navigation

**State**: Agent position on 2D grid
**Tools**: `up`, `down`, `left`, `right`, `get_position`
**Goal**: Navigate to target positions

Location: `examples/directional.py`

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

```bash
# Terminal 1: Start server with directional example
uv run python -m agentenv_mcp.launch --example directional

# Terminal 2: Run tests
uv run python test_env.py
```

## 🔑 Key Design

1. **Generic Core**: Environment doesn't know what tools exist
2. **Pluggable Tools**: Tools injected via factory functions
3. **Shared State**: Tools operate on shared state per instance
4. **Observations from Tools**: Each tool returns observation after modifying state
5. **AgentGym API**: Standard /create, /step, /reset endpoints
6. **Multi-Instance**: Each environment has separate state

## 📖 Similar to

- **agentenv-babyai**: Grid navigation with objects
- **agentenv-wordle**: Word guessing game
- **agentenv-mcp**: Generic framework for MCP tools ← You are here

## 📄 License

MIT License
