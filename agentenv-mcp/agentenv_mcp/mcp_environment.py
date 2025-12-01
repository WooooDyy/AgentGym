"""
MCP-based Environment using MCP Clients.

This environment connects to external MCP servers and uses their tools
as the action space.
"""

from typing import Dict, Any, List
from .mcp_client import MCPClient, MCPClientManager


class MCPBasedEnvironment:
    """
    Environment that uses MCP clients to interact with MCP servers.

    Actions = MCP tool calls
    Observations = MCP tool results
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        task_description: str = "Interact with MCP tools",
        max_steps: int = 50,
    ):
        """
        Initialize MCP-based environment.

        Args:
            mcp_client: MCP client for tool execution
            task_description: Description of the task
            max_steps: Maximum steps per episode
        """
        self.mcp_client = mcp_client
        self.task_description = task_description
        self.max_steps = max_steps

        # Episode state
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.history: List[str] = []

    def reset(self) -> str:
        """Reset environment to initial state."""
        self.current_step = 0
        self.total_reward = 0.0
        self.done = False
        self.history = []

        # Call reset tool if available
        tools = self.mcp_client.list_tools()
        if "reset" in tools:
            self.mcp_client.call_tool("reset")

        # Return initial observation
        obs = f"Task: {self.task_description}\n\n"
        obs += "Environment initialized.\n"
        obs += f"Available actions: {tools}"

        return obs

    def step(self, action: str) -> Dict[str, Any]:
        """
        Execute an action (MCP tool call).

        Args:
            action: Tool name to execute

        Returns:
            Dict with observation, reward, score, done
        """
        if self.done:
            return {
                "observation": "Episode is done. Please reset.",
                "reward": 0.0,
                "score": self.total_reward,
                "done": True,
            }

        self.current_step += 1
        self.history.append(action)

        # Execute tool via MCP client
        observation = self.mcp_client.call_tool(action)

        # Calculate reward (simple: 0.1 for valid action)
        reward = 0.1 if not observation.startswith("Error") else -0.1
        self.total_reward += reward

        # Check max steps
        if self.current_step >= self.max_steps:
            self.done = True

        return {
            "observation": observation,
            "reward": reward,
            "score": self.total_reward,
            "done": self.done,
        }

    def observe(self) -> str:
        """Get current observation without taking action."""
        tools = self.mcp_client.list_tools()
        obs = f"Step {self.current_step}/{self.max_steps}\n"
        obs += f"Available actions: {tools}\n"
        if self.history:
            obs += f"Last action: {self.history[-1]}"
        return obs

    def get_action_space(self) -> List[str]:
        """Get available actions (MCP tools)."""
        return self.mcp_client.list_tools()


class MCPEnvServerV2:
    """
    Multi-instance environment manager for MCP-based environments.

    Uses MCP clients to provide tools from external MCP servers.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize server.

        Args:
            config_path: Path to mcp.json configuration file
        """
        self._max_id = 0
        self.envs: Dict[int, MCPBasedEnvironment] = {}
        self.info: Dict[int, Dict[str, Any]] = {}
        self.ls: list = []

        # MCP client manager
        self.mcp_manager = MCPClientManager(config_path)
        self.default_client_name = None

    def set_mcp_config(self, config_path: str, default_client: str = None):
        """
        Configure MCP clients.

        Args:
            config_path: Path to mcp.json
            default_client: Name of default MCP client to use
        """
        self.mcp_manager = MCPClientManager(config_path)
        self.mcp_manager.initialize_clients()
        self.mcp_manager.connect_all()
        self.default_client_name = default_client

    def create(self) -> Dict[str, int]:
        """Create a new environment instance."""
        try:
            idx = self._max_id
            self._max_id += 1

            self.info[idx] = {"deleted": False, "done": False}
            self.ls.append(idx)

            return {"id": idx}
        except Exception as e:
            return {"error": str(e)}

    def reset(self, env_id: int, data_idx: int = 0) -> Dict[str, Any]:
        """
        Reset environment instance.

        Args:
            env_id: Environment ID
            data_idx: Task/data index

        Returns:
            Dict with observation, reward, score, done
        """
        try:
            # Get MCP client
            if not self.default_client_name:
                # Use first available client
                if self.mcp_manager.clients:
                    self.default_client_name = list(self.mcp_manager.clients.keys())[0]
                else:
                    return {"error": "No MCP clients configured"}

            mcp_client = self.mcp_manager.get_client(self.default_client_name)
            if not mcp_client:
                return {"error": f"MCP client '{self.default_client_name}' not found"}

            # Create environment if it doesn't exist
            if env_id not in self.envs:
                task = f"Complete task {data_idx}"
                self.envs[env_id] = MCPBasedEnvironment(
                    mcp_client=mcp_client, task_description=task, max_steps=50
                )

            # Reset the environment
            observation = self.envs[env_id].reset()

            payload = {
                "observation": observation,
                "reward": 0.0,
                "score": 0.0,
                "done": False,
                "deleted": False,
            }

            self.info[env_id].update(payload)
            return payload

        except Exception as e:
            return {"error": str(e)}

    def step(self, env_id: int, action: str) -> Dict[str, Any]:
        """
        Execute an action in the environment.

        Args:
            env_id: Environment ID
            action: Action to take (tool name)

        Returns:
            Dict with observation, reward, score, done
        """
        try:
            if env_id not in self.envs:
                return {"error": f"Environment {env_id} not found"}

            if self.info[env_id]["deleted"]:
                return {"error": f"Environment {env_id} has been deleted"}

            if self.info[env_id]["done"]:
                return {"error": f"Environment {env_id} is done. Please reset."}

            # Execute step
            result = self.envs[env_id].step(action)

            # Update info
            self.info[env_id].update(result)
            self.info[env_id]["deleted"] = False

            return result

        except Exception as e:
            return {"error": str(e)}

    def observe(self, env_id: int) -> Dict[str, Any]:
        """Get current observation without taking action."""
        try:
            if env_id not in self.envs:
                return {"error": f"Environment {env_id} not found"}

            observation = self.envs[env_id].observe()

            return {
                "observation": observation,
                "reward": self.envs[env_id].total_reward,
                "score": self.envs[env_id].total_reward,
                "done": self.envs[env_id].done,
                "deleted": self.info[env_id].get("deleted", False),
            }

        except Exception as e:
            return {"error": str(e)}

    def close(self, env_id: int) -> bool:
        """Close and cleanup environment."""
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)

            if env_id in self.envs:
                del self.envs[env_id]

            if env_id in self.info:
                self.info[env_id]["deleted"] = True

            print(f"Environment {env_id} closed")
            return True

        except Exception as e:
            print(f"Error closing environment {env_id}: {e}")
            return False

    def __del__(self):
        """Cleanup all environments on deletion."""
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass

        # Close all MCP clients
        self.mcp_manager.close_all()


# Global server instance
server_v2 = MCPEnvServerV2()
