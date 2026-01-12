# -*- coding: utf-8 -*-
"""Docker MCP (Model Context Protocol) HTTP client."""

import json
from typing import Any, Dict, List

import httpx

from ..config import DOCKER_MCP_HOST, DOCKER_MCP_PORT, DOCKER_MCP_URL
from ..utils.logging_config import setup_logging


class DockerMCPClient:
    """
    Client for connecting to MCP server via HTTP (streamableHttp protocol).

    Manages connection to a Docker-hosted MCP server and provides tool execution.

    Attributes:
        tools: List of available tools
        _connected: Connection status
        mcp_url: MCP endpoint URL
        base_url: Base URL for health checks
        health_url: Health check endpoint URL
    """

    def __init__(self) -> None:
        """Initialize the Docker MCP client."""
        self.tools: List[Dict] = []
        self._connected = False
        self.mcp_url = DOCKER_MCP_URL
        self.base_url = f"http://{DOCKER_MCP_HOST}:{DOCKER_MCP_PORT}"
        self.health_url = f"{self.base_url}/healthz"
        self.log = setup_logging("docker-mcp-client")

    async def connect_to_server(self) -> None:
        """Connect to Docker MCP server via HTTP."""
        self.log.info(f"Connecting to Docker MCP server: {self.mcp_url}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check server health first
                try:
                    health_response = await client.get(self.health_url)
                    self.log.info(
                        f"Health check: {self.health_url} -> {health_response.status_code}"
                    )
                except Exception:
                    self.log.warning(
                        f"Could not check health endpoint: {self.health_url}"
                    )

                # MCP protocol headers for streamableHttp
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Version": "2024-11-05",
                    "Protocol-Version": "2024-11-05",
                }

                # JSON-RPC 2.0 format for tools/list
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "params": {},
                    "id": str(hash("tools_list_request"))[:8],
                }

                tools_response = await client.post(
                    self.mcp_url, json=payload, headers=headers
                )

                self.log.info(
                    f"Tools list request to {self.mcp_url} -> {tools_response.status_code}"
                )

                if tools_response.status_code == 200:
                    await self._process_sse_response(tools_response.text)
                    self._connected = True
                elif tools_response.status_code in [400, 404, 405]:
                    # Server is accessible but request format might be wrong
                    self._connected = True
                    self.log.info(
                        f"streamableHttp server accessible (status: {tools_response.status_code}), "
                        f"but could not retrieve tools"
                    )
                else:
                    self.log.warning(
                        f"MCP endpoint unavailable: {self.mcp_url}, status: {tools_response.status_code}"
                    )

        except httpx.ConnectError:
            self.log.warning(f"Cannot connect to Docker MCP server: {self.base_url}")
            self._connected = False
        except httpx.TimeoutException:
            self.log.warning(f"Timeout connecting to Docker MCP server: {self.base_url}")
            self._connected = False
        except Exception as exc:
            self.log.warning(f"Error connecting to Docker MCP server: {exc}")
            self._connected = False

    async def _process_sse_response(self, response_text: str) -> None:
        """
        Process SSE response to extract tools.

        Args:
            response_text: Response text from the server
        """
        try:
            # Handle both direct JSON and SSE-formatted responses
            if response_text.strip().startswith("event: message") or "data: " in response_text:
                # SSE response
                lines = response_text.strip().split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        try:
                            data_content = line[6:]  # Remove 'data: ' prefix
                            sse_data = json.loads(data_content)

                            if (
                                sse_data.get("jsonrpc") == "2.0"
                                and "result" in sse_data
                                and isinstance(sse_data["result"], dict)
                                and "tools" in sse_data["result"]
                            ):
                                self._process_tools(sse_data["result"]["tools"])
                                self.log.info(f"Tools received via SSE: {len(self.tools)}")
                                for tool in self.tools:
                                    self.log.info(
                                        f"  - {tool['function']['name']}: {tool['function']['description']}"
                                    )
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                # Direct JSON response
                data = json.loads(response_text)
                if (
                    data.get("jsonrpc") == "2.0"
                    and "result" in data
                    and isinstance(data["result"], dict)
                    and "tools" in data["result"]
                ):
                    self._process_tools(data["result"]["tools"])
                    self.log.info(f"Tools received via JSON: {len(self.tools)}")
                    for tool in self.tools:
                        self.log.info(
                            f"  - {tool['function']['name']}: {tool['function']['description']}"
                        )
        except Exception as e:
            self.log.warning(f"Error parsing SSE response: {e}")

    def _process_tools(self, tools_list: List[Dict]) -> None:
        """
        Process tools list from server response.

        Args:
            tools_list: List of tool definitions from server
        """
        self.tools = []

        for tool in tools_list:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                },
            }
            self.tools.append(openai_tool)

    async def call_tool(self, name: str, arguments: Dict) -> str:
        """
        Call a Docker MCP server tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        if not self._connected:
            return "[Warning] Docker MCP server unavailable"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Version": "2024-11-05",
                    "Protocol-Version": "2024-11-05",
                }

                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                    "id": str(hash(f"tool_call_{name}"))[:8],
                }

                response = await client.post(self.mcp_url, json=payload, headers=headers)

                if response.status_code == 200:
                    try:
                        if response.text.strip():
                            result = response.json()
                            if isinstance(result, dict):
                                if "result" in result:
                                    return str(result["result"])
                                elif "content" in result:
                                    return str(result["content"])
                                elif "message" in result:
                                    return str(result["message"])
                                else:
                                    return str(result)
                            else:
                                return str(result)
                        else:
                            return "Tool executed successfully (no result)"
                    except Exception:
                        # Server responded with 200 but response is not valid JSON
                        return f"Tool executed successfully, response: {response.text[:200]}..."
                else:
                    return f"[Error] Status code: {response.status_code}, Response: {response.text}"

        except httpx.ConnectError:
            return "[Error] Cannot connect to Docker MCP server"
        except httpx.TimeoutException:
            return "[Error] Tool execution timeout"
        except Exception as exc:
            error_msg = f"Error calling tool {name}: {exc}"
            self.log.error(error_msg)
            return f"[Error] {error_msg}"

    @property
    def available_tools(self) -> List[Dict]:
        """Return list of available tools."""
        return self.tools

    @property
    def connected(self) -> bool:
        """Return connection status."""
        return self._connected
